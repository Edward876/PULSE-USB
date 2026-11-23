"""Single-process localhost server. Run: python app.py --source demo."""
import argparse
import asyncio
import copy
from contextlib import asynccontextmanager
import math
from pathlib import Path
import socket
import struct
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from ppg import PPG
from stress import LiveStress, read_config

ROOT = Path(__file__).parent


class Monitor:
    def __init__(self, source="phone", device="/dev/video10", channel="auto", native_port=None, stress=None):
        self.source, self.device, self.channel = source, device, channel
        self.native_port = native_port
        self.lock = threading.Lock()
        self.analysis_lock = threading.Lock()
        self.engine = PPG(channel)
        self.running = False
        self.shutdown = threading.Event()
        self.last_frame = 0
        self.started = 0
        self.error = ""
        self.phone_connected = False
        self.native_protocol = None
        self.thread = None
        self.native_thread = None
        self.stress = stress or LiveStress(source=source)

    def control(self, start):
        with self.lock:
            self.running = start
            self.engine = PPG(self.channel)
            self.last_frame = 0
            self.started = time.monotonic()
            self.error = ""
            self.stress.control(start)

    def frame(self, frame, timestamp):
        h, w = frame.shape[:2]
        roi = frame[h//5:4*h//5, w//5:4*w//5]
        means = roi.mean(axis=(0, 1))[::-1]  # OpenCV BGR → RGB
        rgb_roi = roi[:, :, ::-1]
        clipped = np.mean((rgb_roi <= 2) | (rgb_roi >= 253), axis=(0, 1))
        self.sample(timestamp, means, clipped)

    def sample(self, timestamp, rgb, clipped, motion=0.0):
        with self.lock:
            if self.running:
                previous = self.engine.samples[-1][0] if self.engine.samples else None
                self.engine.add(timestamp, rgb, clipped, motion)
                if self.engine.samples and self.engine.samples[-1][0] != previous:
                    self.last_frame = time.monotonic()
                    self.stress.add(self.engine.samples[-1])

    def worker(self):
        cap = None
        rng = np.random.default_rng(7)
        try:
            while not self.shutdown.is_set():
                if not self.running:
                    if cap is not None:
                        cap.release()
                        cap = None
                    self.shutdown.wait(0.05)
                    continue
                if self.source == "demo":
                    now = time.monotonic()
                    v = 110 + 4*math.sin(2*math.pi*1.2*now) + rng.normal(0, .12)
                    self.sample(now, [190, v, 60], 0)
                    self.shutdown.wait(1/30)
                else:
                    if cap is None:
                        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ok, frame = cap.read()
                    if not ok:
                        with self.lock:
                            self.error = f"Cannot read {self.device}; check USB bridge and permissions"
                        cap.release()
                        cap = None
                        self.shutdown.wait(0.5)
                        continue
                    with self.lock:
                        self.error = ""
                    self.frame(frame, time.monotonic())
        finally:
            if cap is not None:
                cap.release()

    def _recv_exact(self, conn, size):
        data = bytearray()
        while len(data) < size and not self.shutdown.is_set():
            try:
                chunk = conn.recv(size - len(data))
            except socket.timeout:
                continue
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data) if len(data) == size else None

    def native_worker(self):
        """Receive raw statistics (PPG2/PPG3) or legacy JPEGs over ADB."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(("127.0.0.1", self.native_port))
                server.listen(1)
                server.settimeout(0.5)
                while not self.shutdown.is_set():
                    try:
                        conn, _ = server.accept()
                    except socket.timeout:
                        continue
                    with conn:
                        conn.settimeout(0.5)
                        with self.lock:
                            if self.phone_connected:
                                continue
                            self.phone_connected = True
                            self.error = ""
                        try:
                            protocol = self._recv_exact(conn, 4)
                            if protocol not in {b"PPG1", b"PPG2", b"PPG3"}:
                                with self.lock:
                                    self.error = "Unsupported Android stream protocol"
                                continue
                            with self.lock:
                                self.native_protocol = protocol.decode("ascii")
                            if protocol == b"PPG3":
                                # PPG3 verifies the endpoint before Android starts streaming.
                                conn.sendall(b"\x01")
                            last_timestamp = -1
                            invalid_records = 0
                            while not self.shutdown.is_set():
                                if protocol in {b"PPG2", b"PPG3"}:
                                    record = self._recv_exact(conn, 36)
                                    if record is None:
                                        break
                                    timestamp_ns, red, green, blue, clip_r, clip_g, clip_b, motion = struct.unpack(
                                        ">Qfffffff", record
                                    )
                                    values = np.asarray(
                                        [red, green, blue, clip_r, clip_g, clip_b, motion], dtype=float
                                    )
                                    valid = (
                                        timestamp_ns > last_timestamp
                                        and np.isfinite(values).all()
                                        and np.all((values[:3] >= 0) & (values[:3] <= 255))
                                        and np.all((values[3:6] >= 0) & (values[3:6] <= 1))
                                        and 0 <= motion <= 100
                                    )
                                    if not valid:
                                        with self.lock:
                                            self.error = "Android sent invalid camera statistics"
                                        invalid_records += 1
                                        conn.sendall(b"\x01")
                                        if invalid_records >= 10:
                                            break
                                        continue
                                    invalid_records = 0
                                    last_timestamp = timestamp_ns
                                    # Reply before analysis so UI/DSP work can never stall capture.
                                    conn.sendall(b"\x01")
                                    self.sample(
                                        timestamp_ns / 1_000_000_000,
                                        values[:3],
                                        values[3:6],
                                        motion,
                                    )
                                    continue
                                header = self._recv_exact(conn, 12)
                                if header is None:
                                    break
                                timestamp_ns, length = struct.unpack(">QI", header)
                                if timestamp_ns <= last_timestamp or not 1 <= length <= 200_000:
                                    break
                                payload = self._recv_exact(conn, length)
                                if payload is None:
                                    break
                                last_timestamp = timestamp_ns
                                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if frame is None or frame.shape[0] > 720 or frame.shape[1] > 1280:
                                    break
                                self.frame(frame, timestamp_ns / 1_000_000_000)
                                conn.sendall(b"\x01")
                        except (OSError, struct.error) as exc:
                            with self.lock:
                                self.error = f"Android camera stream ended: {exc}"
                        finally:
                            with self.lock:
                                self.phone_connected = False
                                self.native_protocol = None
                                self.engine = PPG(self.channel)
                                self.last_frame = 0
                                self.stress.disconnect()
        except OSError as exc:
            with self.lock:
                self.error = f"Native Camera2 receiver unavailable on port {self.native_port}: {exc}"

    def snapshot(self, include_samples=False):
        # Serialize estimators, but release the capture lock during NeuroKit.
        # Sensor ingestion must not wait for filtering/template computation.
        with self.analysis_lock:
            with self.lock:
                engine = self.engine
                running = self.running
                if running:
                    analyzer = copy.copy(engine)
                    analyzer.samples = engine.samples.copy()
                    analyzer.bpm_history = engine.bpm_history.copy()
                    analyzer.raw_bpm_history = engine.raw_bpm_history.copy()
            result = analyzer.result() if running else {}
            with self.lock:
                if (running and self.running and self.engine is engine
                        and engine.generation == analyzer.generation):
                    engine.selected_channel = analyzer.selected_channel
                    engine.bpm_history = analyzer.bpm_history
                    engine.raw_bpm_history = analyzer.raw_bpm_history
                    engine.channel_switch_pending = analyzer.channel_switch_pending
                    engine.channel_switch_since = analyzer.channel_switch_since
                    engine.last_estimate_time = analyzer.last_estimate_time
                elif self.running:
                    # Start/Stop/reconnect replaced the engine during analysis.
                    result = self.engine._empty()
                if not self.running:
                    result = dict(bpm=None, quality="Stopped", reason="Press Start measurement", fps=0, waveform=[])
                if self.running and time.monotonic()-self.last_frame > 2:
                    result.update(bpm=None, quality="Waiting", reason=self.error or "Waiting for camera frames",
                                  waveform=[], fps=0, signal_score=None, diagnostics={}, channel_used=None,
                                  pulse_channels=[])
                result.update(source=self.source, running=self.running, channel=self.channel,
                              phone_connected=self.phone_connected, transport=self.native_protocol,
                              capture_error=self.error,
                              elapsed=round(time.monotonic()-self.started, 1) if self.running else 0)
                self.stress.observe_quality(result)
                result["stress"] = self.stress.snapshot()
                if include_samples:
                    rows = (list(analyzer.samples) if running and self.running
                            and self.engine is engine and engine.generation == analyzer.generation else [])
                    latest = rows[-1][0] if rows else 0
                    result["signal_samples"] = {
                        "columns": ["time_seconds", "red", "green", "blue",
                                    "clipped_red", "clipped_green", "clipped_blue", "motion"],
                        "time_origin": "Seconds relative to the newest captured sample",
                        "rows": [[row[0] - latest, *row[1:]] for row in rows],
                    }
                    result["stress_samples"] = {
                        "columns": result["signal_samples"]["columns"],
                        "time_origin": "Seconds relative to newest captured stress sample",
                        "rows": self.stress.report_rows(),
                    }
                return result


monitor = Monitor()


@asynccontextmanager
async def lifespan(app):
    monitor.stress.start_worker()
    if monitor.source != "phone":
        monitor.thread = threading.Thread(target=monitor.worker, daemon=True)
        monitor.thread.start()
    elif monitor.native_port is not None:
        monitor.native_thread = threading.Thread(target=monitor.native_worker, daemon=True)
        monitor.native_thread.start()
    try:
        yield
    finally:
        monitor.shutdown.set()
        monitor.stress.close()
        if monitor.thread:
            monitor.thread.join(timeout=2)
        if monitor.native_thread:
            monitor.native_thread.join(timeout=2)


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.middleware("http")
async def fresh_monitor_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith(("/static/", "/api/")):
        response.headers["Cache-Control"] = "no-store"
    return response


def same_origin(origin, host):
    return origin == f"http://{host}"


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.get("/phone")
def phone():
    return FileResponse(ROOT / "static/phone.html")


@app.get("/api/status")
def status():
    return monitor.snapshot()


@app.get("/api/signal-report")
def signal_report():
    """User-initiated local export for replaying a reported measurement issue."""
    return JSONResponse(monitor.snapshot(include_samples=True), headers={
        "Content-Disposition": 'attachment; filename="pulse-signal-report.json"',
        "Cache-Control": "no-store",
    })


class Control(BaseModel):
    running: bool


@app.post("/api/control")
def control(body: Control, request: Request):
    origin = request.headers.get("origin")
    if origin and not same_origin(origin, request.headers.get("host")):
        raise HTTPException(403, "Local page only")
    monitor.control(body.running)
    return monitor.snapshot()


@app.websocket("/ws")
async def readings(ws: WebSocket):
    if not same_origin(ws.headers.get("origin"), ws.headers.get("host")):
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        while True:
            await ws.send_json(await asyncio.to_thread(monitor.snapshot))
            await asyncio.sleep(.2)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket("/ingest")
async def ingest(ws: WebSocket):
    if monitor.source != "phone" or monitor.phone_connected or not same_origin(ws.headers.get("origin"), ws.headers.get("host")):
        await ws.close(code=1008)
        return
    monitor.phone_connected = True
    await ws.accept()
    last_timestamp = -1
    try:
        while True:
            payload = await ws.receive_bytes()
            # Little-endian capture timestamp in milliseconds, followed by JPEG.
            if not 8 < len(payload) <= 200_000:
                await ws.close(code=1009)
                break
            stamp = float(np.frombuffer(payload[:8], dtype="<f8")[0])/1000
            if not math.isfinite(stamp) or stamp <= last_timestamp:
                await ws.close(code=1008)
                break
            last_timestamp = stamp
            frame = cv2.imdecode(np.frombuffer(payload[8:], dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[0] > 720 or frame.shape[1] > 1280:
                await ws.close(code=1008)
                break
            monitor.frame(frame, stamp)
            await ws.send_json({"running": monitor.running})  # backpressure: one frame in flight
    except WebSocketDisconnect:
        pass
    finally:
        with monitor.lock:
            monitor.phone_connected = False
            monitor.engine = PPG(monitor.channel)
            monitor.last_frame = 0
            monitor.stress.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["phone", "camera", "demo"], default="phone")
    parser.add_argument("--device", default="/dev/video10")
    parser.add_argument("--channel", choices=["auto", "green", "red", "blue"], default="auto")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--native-port", type=int, default=9001,
                        help="ADB-reversed TCP port used by the optional Camera2 Android app")
    parser.add_argument("--stress-bundle", help="Override the pinned trained bundle in stress_model.json")
    parser.add_argument("--stress-python", help="Python environment containing PyTorch and model dependencies")
    parser.add_argument("--no-stress", action="store_true", help="Run only the existing BPM monitor")
    args = parser.parse_args()
    stress = LiveStress(source=args.source)
    if not args.no_stress:
        try:
            stress = LiveStress(read_config(args.stress_bundle, args.stress_python), source=args.source)
        except (OSError, ValueError, KeyError) as exc:
            stress = LiveStress({}, source=args.source)
            stress.load_error = f"Stress configuration could not load: {exc}"
    monitor = Monitor(args.source, args.device, args.channel,
                      args.native_port if args.source == "phone" else None,
                      stress=stress)
    uvicorn.run(app, host="127.0.0.1", port=args.port, ws_max_size=200_000)
