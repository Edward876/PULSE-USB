"""Bounded raw capture and asynchronous, session-safe experimental stress inference."""
from collections import deque
import copy
import json
import logging
import math
from pathlib import Path
import select
import subprocess
import threading
import time

from ppg import PPG

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
INPUT_SECONDS = 64.0
UPDATE_SECONDS = 30.0
CHANNELS = {"red": (1, 4), "green": (2, 5), "blue": (3, 6)}


def read_config(bundle=None, python=None):
    config = json.loads((ROOT / "stress_model.json").read_text())
    path = Path(bundle or config["bundle_dir"]).expanduser()
    interpreter = Path(python or config["python"]).expanduser()
    return dict(bundle_dir=path.resolve() if path.is_absolute() else (ROOT / path).resolve(),
                python=str(interpreter))


class ModelProcess:
    """Persistent child, so torch and model work never run on the capture/UI thread."""
    def __init__(self, config):
        self.config = config
        self.process = None

    def receive(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise TimeoutError("Stress model response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Stress model process exited unexpectedly")
        return json.loads(line)

    def start(self):
        self.process = subprocess.Popen(
            [self.config["python"], "-u", str(ROOT / "stress_worker.py"),
             "--bundle", str(self.config["bundle_dir"])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        reply = self.receive(45)
        if reply.get("type") != "ready":
            raise RuntimeError(reply.get("reason", "Model did not become ready"))
        return reply

    def predict(self, rows, channel):
        self.process.stdin.write(json.dumps({"type": "predict", "rows": rows, "channel": channel}) + "\n")
        self.process.stdin.flush()
        return self.receive(20)

    def close(self):
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()


class LiveStress:
    def __init__(self, config=None, source="phone", *, predictor=None, clock=time.monotonic):
        self.config, self.source, self.predictor, self.clock = config, source, predictor, clock
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.shutdown = threading.Event()
        self.thread = None
        self.process = None
        self.ready = predictor is not None
        self.load_error = ""
        self.threshold = None
        self.generation = 0
        self.running = False
        self.rows = deque(maxlen=24000)
        self.channel = None
        self.alternate_channels = ()
        self.latest_quality = None
        self.quality_at = 0.0
        self.poor_since = None
        self.last_frame = None
        self.last_attempt = None
        self.last_estimate = None
        self.last_rejection = None
        self.retry_seconds = UPDATE_SECONDS
        self.bad_capture_since = None
        self.reset_count = 0
        self.last_reset_reason = None
        self.in_flight = False
        self.reason = "Press Start measurement"

    def start_worker(self):
        if self.source == "demo" or self.config is None or self.load_error or self.thread is not None:
            return
        self.thread = threading.Thread(target=self._worker, name="pulse-stress", daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            process = ModelProcess(self.config)
            with self.lock:
                self.process = process
            info = process.start()
            with self.lock:
                self.ready = True
                self.threshold = float(info["threshold"])
            while not self.shutdown.is_set():
                self.wake.wait(.25)
                self.wake.clear()
                self.poll()
        except Exception as exc:
            if not self.shutdown.is_set():
                log.exception("Stress inference worker stopped")
            with self.lock:
                self.ready = False
                self.in_flight = False
                self.load_error = str(exc)
        finally:
            if self.process:
                self.process.close()

    def close(self):
        self.shutdown.set()
        self.wake.set()
        if self.process:
            self.process.close()
        if self.thread:
            self.thread.join(timeout=3)

    def _reset_segment(self, reason, *, keep_channel=False):
        self.rows.clear()
        self.generation += 1
        self.last_attempt = None
        self.last_rejection = None
        self.retry_seconds = UPDATE_SECONDS
        self.bad_capture_since = None
        self.reset_count += 1
        self.last_reset_reason = reason
        if not keep_channel:
            self.channel = None
            self.alternate_channels = ()
        self.reason = reason

    def control(self, running):
        with self.lock:
            self.running = running
            self._reset_segment("Collecting a usable pulse" if running else "Press Start measurement")
            self.latest_quality = None
            self.quality_at = 0.0
            self.poor_since = None
            self.last_frame = None
            self.last_estimate = None
            self.reset_count = 0
            self.last_reset_reason = None
        self.wake.set()

    def disconnect(self, reason="Camera disconnected; reconnect and collect a new window"):
        with self.lock:
            self._reset_segment(reason)
            self.latest_quality = None
            self.last_frame = None

    def observe_quality(self, result):
        with self.lock:
            if not self.running or self.source == "demo" or (self.config is None and self.predictor is None):
                return
            now = self.clock()
            quality = result.get("quality")
            self.latest_quality, self.quality_at = quality, now
            if quality == "Good":
                self.poor_since = None
                chosen = result.get("channel_used")
                if chosen in CHANNELS:
                    if self.channel is None:
                        # During acquisition we stored every plausible RGB row.
                        # Only credit the continuous, correctly exposed suffix
                        # for the specific color that actually passed pulse checks.
                        vi, ci = CHANNELS[chosen]
                        last_bad = next((r[0] for r in reversed(self.rows)
                                         if r[vi] < 10 or r[ci] > .30 or r[7] > 2), None)
                        while self.rows and last_bad is not None and self.rows[0][0] <= last_bad:
                            self.rows.popleft()
                    # Every row retains ALL RGB values. Choose one column for the
                    # entire next job; a new preference does not splice waveforms.
                    self.channel = chosen
                    self.alternate_channels = tuple(
                        name for name in result.get("pulse_channels", ())
                        if name in CHANNELS and name != chosen
                    )
                    if self.last_estimate is None and self.last_rejection is None:
                        self.reason = "Collecting the latest 64 seconds of camera data"
            elif quality in {"Poor", "Waiting"}:
                if self.poor_since is None:
                    self.poor_since = now
                # Brief DSP quality changes do not destroy raw camera history.
                # Real acquisition breaks are handled from capture timestamps.
        self.wake.set()

    def add(self, row):
        """Accept only validated camera rows. All checks here are cheap, no DSP/ML."""
        with self.lock:
            if (not self.running or self.source == "demo" or self.load_error or
                    (self.config is None and self.predictor is None)):
                return
            now = self.clock()
            self.last_frame = now
            if self.rows:
                delta = row[0] - self.rows[-1][0]
                if delta <= 0:
                    return
                if delta > .2501:
                    self._reset_segment("A capture gap interrupted the 64-second window", keep_channel=True)
            if self.channel is None:
                usable_light = any(row[vi] >= 10 and row[ci] <= .30 for vi, ci in CHANNELS.values())
            else:
                vi, ci = CHANNELS[self.channel]
                usable_light = row[vi] >= 10 and row[ci] <= .30
            if not usable_light or row[7] > 2:
                if not self.rows:
                    return
                if self.bad_capture_since is None:
                    self.bad_capture_since = row[0]
                if row[0] - self.bad_capture_since >= .5:
                    self._reset_segment("Finger contact or exposure was lost; collecting again")
                    return
            else:
                self.bad_capture_since = None
            self.rows.append(tuple(row))
            # Limit provisional history to one BPM analysis window. A long
            # unverified warm-up must never qualify as a whole stress recording.
            limit = PPG.ANALYSIS_SECONDS if self.channel is None else INPUT_SECONDS
            while len(self.rows) > 2 and self.rows[1][0] <= row[0] - limit:
                self.rows.popleft()
        self.wake.set()

    def poll(self):
        """Called only by the model worker; separate method enables deterministic tests."""
        with self.lock:
            now = self.clock()
            if (not self.running or not self.ready or self.in_flight or self.source == "demo"
                    or self.channel is None or self.latest_quality != "Good" or now - self.quality_at > 5
                    or self.last_frame is None or now - self.last_frame > 2
                    or len(self.rows) < 2 or self.rows[-1][0] - self.rows[0][0] < INPUT_SECONDS
                    or (self.last_attempt is not None and self.rows[-1][0] - self.last_attempt < self.retry_seconds)):
                return
            rows = list(self.rows)
            channels = (self.channel, *self.alternate_channels)
            generation, end = self.generation, rows[-1][0]
            self.in_flight = True
            self.last_attempt = end
        try:
            for channel in channels:
                if self.predictor is not None:
                    reply = {"type": "result", "result": self.predictor(rows, channel)}
                else:
                    reply = self.process.predict(rows, channel)
                if reply.get("type") != "rejected":
                    break
                with self.lock:
                    if self.generation != generation:
                        break
        except Exception as exc:
            reply = {"type": "error", "reason": str(exc)}
        with self.lock:
            self.in_flight = False
            if not self.running or self.generation != generation:
                return  # Stop, reconnect or bad capture invalidated this job.
            if reply.get("type") == "result":
                value = reply["result"]
                score, threshold = float(value["stress_score"]), float(value["threshold"])
                if not (math.isfinite(score) and 0 <= score <= 1 and math.isfinite(threshold) and 0 < threshold < 1):
                    self.ready = False
                    self.load_error = "Model returned an invalid score"
                    self._reset_segment(self.load_error)
                    return
                self.last_estimate = {"score": score, "threshold": threshold,
                    "label": value["label"], "channel": channel,
                    "generation": generation,
                    "computed_at": self.clock(), "window_end": end,
                    "signal_diagnostics": value.get("signal_diagnostics", {}),
                    "capture_diagnostics": value.get("capture_diagnostics", {})}
                self.last_rejection = None
                self.retry_seconds = UPDATE_SECONDS
                self.reason = "Experimental estimate from the most recent 64-second window"
            else:
                self.reason = reply.get("reason", "Model could not read this window")
                if reply.get("type") == "error":
                    self.ready = False
                    self.load_error = self.reason
                    self._reset_segment(self.reason, keep_channel=True)
                else:
                    # Retain history: the bad part ages out naturally. Retry a
                    # new rolling window rather than restarting the whole minute.
                    self.last_rejection = self.reason
                    self.retry_seconds = 5.0

    def snapshot(self):
        with self.lock:
            now = self.clock()
            duration = self.rows[-1][0] - self.rows[0][0] if len(self.rows) > 1 else 0.0
            estimate = copy.deepcopy(self.last_estimate)
            fresh = bool(estimate and self.running and self.rows
                         and self.latest_quality == "Good" and now - self.quality_at <= 5
                         and self.last_rejection is None
                         and self.last_frame is not None and now - self.last_frame <= 2
                         and now - estimate["computed_at"] <= 45
                         and estimate["generation"] == self.generation) if estimate else False
            status = "Collecting"
            reason = self.reason
            if self.source == "demo":
                status, reason = "Demo", "Stress inference is disabled for simulated pulse data"
            elif self.config is None and self.predictor is None:
                status, reason = "Disabled", "No stress model configured"
            elif self.load_error:
                status, reason = "Unavailable", self.load_error
            elif not self.ready:
                status, reason = "Loading", "Loading the trained model in its Python environment"
            elif not self.running:
                status, reason = "Stopped", "Press Start measurement"
            elif self.last_frame is None or now - self.last_frame > 2:
                status, reason = "Waiting", "Waiting for live camera frames"
            elif fresh:
                status = "Updating" if self.in_flight else "Ready"
            elif estimate:
                status = "Stale"
                reason = self.reason if self.latest_quality == "Good" else "Pulse quality changed; previous estimate is stale"
            elif self.in_flight:
                status, reason = "Analyzing", "Analyzing the first complete window"
            elif self.last_rejection:
                status, reason = "Retrying", self.last_rejection
            elif self.channel is None:
                status, reason = "Acquiring pulse", "Waiting for the first reliable BPM before accepting a stress window"
            elif duration >= INPUT_SECONDS and self.latest_quality != "Good":
                status, reason = "Checking signal", "64 seconds collected; waiting for the pulse signal to settle"
            # Last result is always explicitly historical when not fresh.
            return {"status": status, "reason": reason, "experimental": True,
                "score": estimate["score"] if fresh else None,
                "label": estimate["label"] if fresh else None,
                "threshold": estimate["threshold"] if estimate else self.threshold,
                "age_seconds": round(now - estimate["computed_at"], 1) if estimate else None,
                "previous_estimate": estimate if estimate and not fresh else None,
                "channel": estimate["channel"] if fresh else self.channel,
                "collected_seconds": math.floor(min(INPUT_SECONDS, duration) * 10) / 10,
                "required_seconds": INPUT_SECONDS, "update_seconds": UPDATE_SECONDS,
                "next_update_seconds": round(max(0, self.retry_seconds - (self.rows[-1][0] - self.last_attempt)), 1)
                    if self.rows and self.last_attempt is not None else None,
                "signal_diagnostics": estimate["signal_diagnostics"] if fresh else {},
                "capture_diagnostics": estimate.get("capture_diagnostics", {}) if fresh else {},
                "reset_count": self.reset_count, "last_reset_reason": self.last_reset_reason,
                "model": "WESAD PPG · Inception v2", "phone_validation": "not established"}

    def report_rows(self):
        with self.lock:
            rows = list(self.rows)
            latest = rows[-1][0] if rows else 0
            return [[row[0] - latest, *row[1:]] for row in rows]
