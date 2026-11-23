"""Timestamp-aware fingertip PPG estimator for an educational prototype."""
from collections import deque
import logging
import warnings

import neurokit2 as nk
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import correlate, find_peaks

logger = logging.getLogger(__name__)


class PPG:
    """Estimate pulse only when independent time/frequency checks agree."""

    CHANNELS = {"red": (1, 4), "green": (2, 5), "blue": (3, 6)}
    WINDOW_SECONDS = 24
    WARMUP_SECONDS = 10
    ANALYSIS_SECONDS = 8

    def __init__(self, channel="auto"):
        if channel not in {"auto", *self.CHANNELS}:
            raise ValueError(f"Unsupported channel: {channel}")
        self.channel = channel
        self.samples = deque(maxlen=4800)
        self.selected_channel = None
        self.bpm_history = deque(maxlen=20)
        self.raw_bpm_history = deque(maxlen=256)
        self.channel_switch_pending = None
        self.channel_switch_since = None
        self.last_estimate_time = None
        self.generation = 0

    def add(self, t, rgb, clipped=0.0, motion=0.0):
        rgb = np.asarray(rgb, dtype=float).reshape(-1)
        clipping = np.asarray(clipped, dtype=float).reshape(-1)
        if clipping.size == 1:
            clipping = np.repeat(clipping, 3)
        values = np.r_[t, rgb, clipping, motion]
        if rgb.size != 3 or clipping.size != 3 or not np.isfinite(values).all():
            return
        if np.any(rgb < 0) or np.any(rgb > 255) or np.any(clipping < 0) or np.any(clipping > 1):
            return
        if self.samples and t <= self.samples[-1][0]:
            return
        if self.samples and t - self.samples[-1][0] > 0.5:
            self.samples.clear()
            self.bpm_history.clear()
            self.raw_bpm_history.clear()
            self.selected_channel = None
            self.channel_switch_pending = None
            self.channel_switch_since = None
            self.last_estimate_time = None
            self.generation += 1
        self.samples.append((float(t), *rgb, *clipping, max(0.0, float(motion))))
        while self.samples and t - self.samples[0][0] > self.WINDOW_SECONDS:
            self.samples.popleft()

    @staticmethod
    def _empty():
        return dict(
            bpm=None,
            quality="Warming up",
            reason="Collecting 10 seconds of steady fingertip data",
            fps=0,
            waveform=[],
            seconds=0,
            channel_used=None,
            signal_score=None,
            diagnostics={},
            processing_method="NeuroKit2 / Elgendi / template match",
            pipeline_version="fingertip-quality-v4",
        )

    @staticmethod
    def _spectral_peak(y, fs):
        nfft = 1 << int(np.ceil(np.log2(max(256, len(y) * 8))))
        spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y)), n=nfft)) ** 2
        frequencies = np.fft.rfftfreq(nfft, 1 / fs)
        band = (frequencies >= 0.65) & (frequencies <= 3.5)
        indices = np.flatnonzero(band)
        peak_index = indices[np.argmax(spectrum[band])]
        dominant = float(frequencies[peak_index])
        nearby = band & (np.abs(frequencies - dominant) <= 0.16)
        concentration = float(spectrum[nearby].sum() / max(spectrum[band].sum(), 1e-12))
        return dominant * 60, concentration, spectrum, frequencies

    @staticmethod
    def _refine_peak_positions(y, peaks):
        """Three-point parabolic timing refinement; does not recover lost data."""
        positions = np.asarray(peaks, dtype=float).copy()
        for i, peak in enumerate(peaks):
            if 0 < peak < len(y) - 1:
                left, center, right = y[peak - 1:peak + 2]
                curvature = left - 2 * center + right
                if curvature < -1e-12:
                    offset = 0.5 * (left - right) / curvature
                    positions[i] += np.clip(offset, -0.5, 0.5)
        return positions

    @staticmethod
    def _peak_rate(y, x, fs, scale):
        # Elgendi et al. (2013): moving averages and adaptive thresholding.
        # Do not correct/insert beats: synthetic corrections must not imply that
        # missing timing information was measured by this camera.
        # At ~30 FPS use a 140 ms peak window and 250 ms refractory period;
        # these are explicit camera adaptations to the published defaults.
        _, info = nk.ppg_peaks(y, sampling_rate=fs, method="elgendi",
                               correct_artifacts=False, mindelay=0.25,
                               peakwindow=0.14)
        peaks = np.asarray(info["PPG_Peaks"], dtype=int)
        if len(peaks) < 5:
            return None, 1.0, peaks
        positions = PPG._refine_peak_positions(y, peaks)
        peak_times = np.interp(positions, np.arange(len(x)), x)
        # Follow the last few beats rather than the median of 24 seconds.
        intervals = np.diff(peak_times)[-5:]
        plausible = intervals[(intervals >= 0.30) & (intervals <= 1.50)]
        if len(plausible) < 4:
            return None, 1.0, peaks
        median = float(np.median(plausible))
        mad = float(np.median(np.abs(plausible - median)))
        tolerance = max(0.20 * median, 3.5 * mad)
        accepted = plausible[np.abs(plausible - median) <= tolerance]
        if len(accepted) < 3:
            return None, 1.0, peaks
        interval_cv = float(np.std(accepted) / max(np.mean(accepted), 1e-9))
        return 60.0 / float(np.mean(accepted)), interval_cv, peaks

    @staticmethod
    def _autocorrelation_rate(y, fs, preferred_bpm=None):
        centered = y - np.mean(y)
        acf = correlate(centered, centered, mode="full", method="fft")[len(centered) - 1:]
        if acf[0] <= 1e-12:
            return None, 0.0
        acf /= acf[0]
        minimum = max(1, int(fs * 0.30))
        maximum = min(len(acf) - 1, int(fs * 1.50))
        section = acf[minimum:maximum + 1]
        candidates, _ = find_peaks(section, distance=max(1, int(fs * 0.16)), prominence=0.04)
        if len(candidates) == 0:
            lag = minimum + int(np.argmax(section))
            refined = PPG._refine_peak_positions(acf, [lag])[0]
            return 60.0 * fs / refined, float(acf[lag])
        lags = minimum + candidates
        strengths = acf[lags]
        viable = strengths >= max(0.18, 0.90 * float(np.max(strengths)))
        lags = lags[viable]
        strengths = strengths[viable]
        if not len(lags):
            return None, 0.0
        refined = PPG._refine_peak_positions(acf, lags)
        bpms = 60.0 * fs / refined
        if preferred_bpm is None:
            selected = int(np.argmax(strengths))
        else:
            distance = np.abs(bpms - preferred_bpm)
            selected = int(np.argmin(distance - 2.0 * strengths))
        return float(bpms[selected]), float(strengths[selected])

    @staticmethod
    def _pulse_quality(y, peaks):
        """Align observed cycles in phase before NeuroKit template comparison.

        Different beat durations are not in themselves poor morphology. Use 64
        phase points per observed cycle, not a fictitious measured sample rate.
        This also avoids NeuroKit 0.2.12's variable-length epoch pivot failure at
        some fractional cycle durations (all cycles were dropped as NaN).
        No normalized positions are ever used to calculate BPM or intervals.
        """
        positions = PPG._refine_peak_positions(y, peaks)[-7:]
        if len(positions) < 5:
            return 0.0
        phase_points = 64
        pulses = [np.interp(np.linspace(left, right, phase_points, endpoint=False),
                            np.arange(len(y)), y)
                  for left, right in zip(positions[:-1], positions[1:])]
        aligned = np.concatenate(pulses)
        phase_peaks = np.arange(0, len(aligned), phase_points)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            quality = np.asarray(nk.ppg_quality(
                aligned, peaks=phase_peaks, sampling_rate=phase_points,
                method="templatematch"))
        interior = quality[phase_points:-phase_points]
        if not interior.size or not np.isfinite(interior).all():
            return 0.0
        return float(np.clip(np.median(interior), 0, 1))

    @staticmethod
    def _support_tolerance(bpm):
        return max(5.0, 0.07 * (bpm or 0))

    def _select_channel(self, candidates, timestamp):
        """Prefer usable channels; require sustained improvement to switch."""
        usable = [c for c in candidates if not c["failed_checks"]]
        pool = usable or candidates
        best = max(pool, key=lambda c: c["score"])
        current = next((c for c in usable if c["name"] == self.selected_channel), None)
        if current is not None and best["name"] != current["name"]:
            if best["score"] >= 1.4 * current["score"]:
                if self.channel_switch_pending != best["name"]:
                    self.channel_switch_pending = best["name"]
                    self.channel_switch_since = timestamp
                if timestamp - self.channel_switch_since < 2.0:
                    return current
            else:
                self.channel_switch_pending = None
                self.channel_switch_since = None
                return current
        self.channel_switch_pending = None
        self.channel_switch_since = None
        self.selected_channel = best["name"]
        return best

    def _stabilize_bpm(self, timestamp, bpm):
        """Median of 1.2 s of supported estimates, then a 1.5 s low-pass.

        This is a rolling displayed rate. Beat timing remains unmodified in
        diagnostics; neither this median nor this smoother is an HRV input.
        Invalid windows never call this method, so an old value is not exposed
        as a fresh measurement while capture is poor.
        """
        if self.last_estimate_time == timestamp:
            return self.bpm_history[-1][1]
        if self.bpm_history and timestamp - self.bpm_history[-1][0] > 2.0:
            self.raw_bpm_history.clear()
            self.bpm_history.clear()
        self.raw_bpm_history.append((timestamp, bpm))
        while self.raw_bpm_history and timestamp - self.raw_bpm_history[0][0] > 1.2:
            self.raw_bpm_history.popleft()
        target = float(np.median([value for _, value in self.raw_bpm_history]))
        if self.bpm_history:
            stamp, previous = self.bpm_history[-1]
            alpha = 1 - np.exp(-(timestamp - stamp) / 1.5)
            target = previous + alpha * (target - previous)
        self.bpm_history.append((timestamp, target))
        self.last_estimate_time = timestamp
        return target

    def _channel_candidate(self, a, t, grid, valid, fs, name):
        index, clipping_index = self.CHANNELS[name]
        recent = a[t > t[-1] - 2]
        brightness = float(np.mean(recent[:, index]))
        clipping = float(np.mean(recent[:, clipping_index]))
        if brightness < 10 or clipping > 0.30:
            return None

        raw = np.interp(grid, t, a[:, index])
        if len(raw) < 3 * fs:
            return None
        local_median = median_filter(raw, size=3, mode="nearest")
        residual = raw - local_median
        residual_mad = float(np.median(np.abs(residual - np.median(residual))))
        # Include ordinary inter-frame pulse slope so a sharp systolic upstroke
        # at camera FPS is not mistaken for an isolated impulse artifact.
        typical_step = float(np.median(np.abs(np.diff(raw))))
        spike_limit = max(0.35, 7.0 * residual_mad, 4.0 * typical_step)
        spikes = np.abs(residual) > spike_limit
        spike_fraction = float(np.mean(spikes[grid >= grid[-1] - 2]))
        cleaned = raw.copy()
        cleaned[spikes] = local_median[spikes]

        # Published Elgendi cleaning uses a 0.5–8 Hz Butterworth filter.
        # Keep pulse morphology rather than narrowing the waveform to HR alone.
        y = nk.ppg_clean(cleaned, sampling_rate=fs, method="elgendi")[valid]
        if not np.isfinite(y).all():
            raise ValueError("Non-finite cleaned waveform")
        x = grid[valid]
        scale = float(np.std(y))
        if scale < 1e-6:
            return None
        # Keep the full filtered window for display; estimate rate/quality only
        # from its recent part so old resting beats cannot dominate a new rate.
        analysis = x >= x[-1] - self.ANALYSIS_SECONDS
        rate_y, rate_x = y[analysis], x[analysis]
        spectral_y = rate_y[rate_x >= rate_x[-1] - 6]
        fft_bpm, concentration, spectrum, frequencies = self._spectral_peak(spectral_y, fs)
        peak_bpm, interval_cv, peaks = self._peak_rate(rate_y, rate_x, fs, scale)
        latest_peak_age = float(t[-1] - rate_x[peaks[-1]]) if len(peaks) else 99.0
        template_quality = 0.0
        if peak_bpm is not None:
            template_quality = self._pulse_quality(rate_y, peaks)

        # A strong second harmonic can be the largest FFT peak. Beat timing identifies
        # the fundamental and lets the frequency estimate follow it when supported.
        if peak_bpm is not None and abs(fft_bpm - 2 * peak_bpm) < 10:
            fundamental = peak_bpm / 60
            fundamental_power = float(np.interp(fundamental, frequencies, spectrum))
            harmonic_power = float(np.interp(2 * fundamental, frequencies, spectrum))
            if fundamental_power >= 0.08 * max(harmonic_power, 1e-12):
                fft_bpm /= 2

        preferred = peak_bpm if peak_bpm is not None else fft_bpm
        acf_bpm, acf_strength = self._autocorrelation_rate(spectral_y, fs, preferred)

        # Some fingertip pulses have a dicrotic second lobe large enough for both
        # peak finding and the FFT to report twice the true rate. Autocorrelation
        # still exposes the full beat period, so use it as a harmonic tie-breaker.
        if (
            peak_bpm is not None
            and acf_bpm is not None
            and acf_strength >= 0.28
            and abs(peak_bpm - 2 * acf_bpm) < 12
            and abs(fft_bpm - 2 * acf_bpm) < 12
        ):
            fundamental_power = float(np.interp(acf_bpm / 60, frequencies, spectrum))
            harmonic_power = float(np.interp(acf_bpm / 30, frequencies, spectrum))
            # A changing rate also creates two correlation periods. Only halve
            # actual peak timing when alternating large/small lobes support a
            # dicrotic double-count, rather than blindly following the ACF.
            amplitudes = rate_y[peaks[-8:]]
            pairs = amplitudes[:len(amplitudes) // 2 * 2].reshape(-1, 2)
            contrasts = pairs[:, 0] - pairs[:, 1]
            alternating = (len(contrasts) >= 3
                           and abs(float(np.median(contrasts))) > 0.35 * scale
                           and max(np.mean(contrasts > 0), np.mean(contrasts < 0)) >= 0.75)
            if alternating and fundamental_power >= 0.08 * max(harmonic_power, 1e-12):
                peak_bpm /= 2
                fft_bpm /= 2
        estimates = [value for value in (peak_bpm, fft_bpm, acf_bpm) if value is not None]
        agreement = float(max(estimates) - min(estimates)) if len(estimates) >= 2 else 99.0
        # FFT/ACF describe a window average, whereas recent intervals follow
        # changing rates. Require one corroborator, not all three at once.
        support = (min(abs(peak_bpm - value) for value in (fft_bpm, acf_bpm)
                       if value is not None) if peak_bpm is not None else 99.0)

        periodicity = min(1.0, concentration / 0.55) * min(1.0, acf_strength / 0.65)
        artifact_penalty = max(0.0, 1.0 - spike_fraction / 0.10)
        amplitude = min(1.0, np.log1p(scale / 0.04) / 3.0)
        score = periodicity * artifact_penalty * amplitude * template_quality
        # Frame values are ROI means, so real changes can be a fraction of
        # one intensity unit. Estimate frame noise from the original samples:
        # the second difference suppresses slow drift and a smooth pulse.
        tail = recent[:, index]
        second_diff = np.diff(tail, n=2)
        frame_noise = float(np.median(np.abs(second_diff - np.median(second_diff)))
                            / (0.67449 * np.sqrt(6)))
        low, high = np.percentile(tail, [10, 90])
        pulse_span = float(high - low)
        tolerance = self._support_tolerance(peak_bpm)
        supported = (peak_bpm is not None and
                     ((concentration >= 0.40 and abs(peak_bpm - fft_bpm) <= tolerance)
                      or (acf_bpm is not None and acf_strength >= 0.50
                          and abs(peak_bpm - acf_bpm) <= tolerance)))
        small = scale < 0.06 or pulse_span < 0.12
        resolved = (pulse_span >= max(0.025, 6 * frame_noise)
                    and template_quality >= 0.90 and supported)
        agreement_score = max(0.0, 1.0 - support / 20.0)
        light_score = max(0.0, 1.0 - clipping / 0.20)
        evidence_score = 100 * (
            0.20 * min(1.0, concentration / 0.50) + 0.20 * min(1.0, acf_strength / 0.60)
            + 0.35 * template_quality + 0.15 * agreement_score + 0.10 * light_score
        )
        checks = {
            "weak": small and not resolved,
            "spikes": spike_fraction > 0.08,
            "few_beats": peak_bpm is None,
            "periodicity": concentration < 0.28 and acf_strength < 0.28,
            "disagreement": support > tolerance,
            "template": template_quality < 0.80,
            "stale_beats": latest_peak_age > max(2.5, 120 / (peak_bpm or 40)),
            "evidence": evidence_score < 62,
        }
        rates, weights = [], []
        if peak_bpm is not None:
            rates, weights = [peak_bpm], [0.6]
            for rate, weight in ((fft_bpm, 0.3), (acf_bpm, 0.1)):
                if rate is not None and abs(rate - peak_bpm) <= tolerance:
                    rates.append(rate)
                    weights.append(weight)
        fused_bpm = float(np.average(rates, weights=weights)) if rates else None
        checks["rate_range"] = fused_bpm is not None and not 40 <= fused_bpm <= 200
        return dict(
            name=name,
            x=x,
            y=y,
            scale=scale,
            brightness=brightness,
            clipping=clipping,
            spike_fraction=spike_fraction,
            concentration=concentration,
            peak_bpm=peak_bpm,
            fft_bpm=fft_bpm,
            acf_bpm=acf_bpm,
            acf_strength=acf_strength,
            interval_cv=interval_cv,
            agreement=agreement,
            support=support,
            peaks=peaks,
            score=float(score),
            template_quality=template_quality,
            analysis_seconds=float(rate_x[-1] - rate_x[0]),
            latest_peak_age=latest_peak_age,
            pulse_span=pulse_span,
            frame_noise=frame_noise,
            low_amplitude=small,
            fused_bpm=fused_bpm,
            evidence_score=evidence_score,
            failed_checks=[name for name, failed in checks.items() if failed],
        )

    def result(self):
        out = self._empty()
        if len(self.samples) < 10:
            return out
        a = np.asarray(self.samples)
        t = a[:, 0]
        duration = float(t[-1] - t[0])
        if duration <= 0:
            return out
        timing_t = t[t >= t[-1] - self.ANALYSIS_SECONDS]
        dt = np.diff(timing_t)
        fs = float((len(timing_t) - 1) / (timing_t[-1] - timing_t[0]))
        jitter = float(np.std(dt) / max(np.mean(dt), 1e-9))
        recent = a[t > t[-1] - 2]
        motion = float(np.percentile(recent[:, 7], 90))
        out.update(fps=round(fs, 1), seconds=round(duration, 1))
        if duration < 3:
            return out
        if fs < 17 or np.max(dt) > 0.2 or jitter > 0.55:
            out.update(
                quality="Poor",
                reason="Need at least 17 FPS and stable timing; target 30 FPS and select a lower camera resolution",
                diagnostics={"frame_jitter": round(jitter, 3)},
            )
            return out
        if motion > 0.65:
            out.update(
                quality="Poor",
                reason="Phone movement detected; place it on a surface and keep the finger still",
                diagnostics={"motion": round(motion, 3)},
            )
            return out

        # NeuroKit expects a sampling rate, not irregular camera arrival times.
        # Rounding changes the grid only; it does not improve timing resolution.
        analysis_fs = int(round(fs))
        grid = np.arange(t[0], t[-1], 1 / analysis_fs)
        if len(grid) < 40:
            return out
        valid = (grid >= grid[0] + 0.7) & (grid <= grid[-1] - 0.6)
        if np.count_nonzero(valid) < 30:
            return out

        names = self.CHANNELS if self.channel == "auto" else (self.channel,)
        candidates = []
        processing_errors = []
        for name in names:
            try:
                candidate = self._channel_candidate(a, t, grid, valid, analysis_fs, name)
                if candidate is not None:
                    candidates.append(candidate)
            except (ValueError, IndexError, ZeroDivisionError, RuntimeError,
                    FloatingPointError) as exc:
                logger.debug("NeuroKit processing failed for %s: %s", name, exc, exc_info=True)
                processing_errors.append(name)
        if not candidates:
            channel_brightness = np.mean(recent[:, 1:4], axis=0)
            channel_clipping = np.mean(recent[:, 4:7], axis=0)
            least_clipped = int(np.argmin(channel_clipping))
            out.update(
                quality="Poor",
                reason=("PPG processing could not resolve pulses; hold still and collect a fresh window"
                        if processing_errors else
                        "No usable pulse channel: signal is flat, dark or clipped; reposition the finger and let exposure adjust"),
                diagnostics={
                    "brightness": round(float(channel_brightness[least_clipped]), 1),
                    "clipping_percent": round(float(100 * channel_clipping[least_clipped]), 1),
                    "motion": round(motion, 3),
                    "frame_jitter": round(jitter, 3),
                },
            )
            return out
        best = self._select_channel(candidates, float(t[-1]))

        x, y, scale = best["x"], best["y"], best["scale"]
        normalized = y / max(scale, 0.01)
        out["waveform"] = [
            [round(float(tx - t[-1]), 3), round(float(value), 3)]
            for tx, value in zip(x, normalized) if tx >= t[-1] - 12
        ]
        out["channel_used"] = best["name"]
        out["diagnostics"] = {
            "neurokit_version": nk.__version__,
            "template_quality": round(best["template_quality"], 3),
            "brightness": round(best["brightness"], 1),
            "clipping_percent": round(100 * best["clipping"], 1),
            "motion": round(motion, 3),
            "frame_jitter": round(jitter, 3),
            "peak_bpm": None if best["peak_bpm"] is None else round(best["peak_bpm"], 1),
            "fft_bpm": round(best["fft_bpm"], 1),
            "autocorrelation_bpm": None if best["acf_bpm"] is None else round(best["acf_bpm"], 1),
            "method_spread_bpm": round(best["agreement"], 1),
            "support_error_bpm": round(best["support"], 1),
            "analysis_seconds": round(best["analysis_seconds"], 1),
            "interval_cv": round(best["interval_cv"], 3),
            "quality_method": "phase-normalized NeuroKit template",
            "latest_peak_age_seconds": round(best["latest_peak_age"], 2),
            "recent_pulse_span": round(best["pulse_span"], 4),
            "frame_noise": round(best["frame_noise"], 4),
            "low_amplitude": best["low_amplitude"],
        }
        if duration < self.WARMUP_SECONDS:
            return out

        failed = best["failed_checks"]
        out["diagnostics"].update(failed_checks=failed, evidence_score=round(best["evidence_score"]))
        if failed:
            if failed == ["disagreement"]:
                out.update(quality="Tracking", reason="Pulse estimates are adjusting; waiting for recent beats to agree")
                return out
            if "spikes" in failed:
                reason = "Abrupt finger or lighting changes detected; reduce pressure changes and hold still"
            elif "few_beats" in failed:
                reason = "Too few reliable pulse peaks; reposition the fingertip over the illuminated lens"
            elif "stale_beats" in failed:
                reason = "No recent reliable pulse detected; check fingertip contact and illumination"
            elif "disagreement" in failed:
                reason = "Pulse estimators disagree; keep still and wait for a cleaner 10-second window"
            elif "template" in failed:
                reason = "Pulse shapes are inconsistent; hold the finger gently and keep illumination steady"
            elif "periodicity" in failed:
                reason = "The waveform is not sufficiently periodic; adjust the finger and illumination"
            elif "weak" in failed:
                reason = "Small camera changes do not yet resolve a reliable pulse; keep light contact over the lit lens"
            elif "rate_range" in failed:
                reason = "Estimated rate is outside the supported 40–200 BPM range"
            else:
                reason = "Pulse evidence is inconsistent across checks; collecting more beats"
            out.update(quality="Poor", reason=reason)
            return out

        # Fuse only corroborating estimates, retaining recent beat timing as
        # the largest weight. Unsupported frequencies cannot drag the display.
        bpm = best["fused_bpm"]
        out["diagnostics"]["fused_bpm"] = round(bpm, 1)
        stable_bpm = self._stabilize_bpm(float(t[-1]), bpm)
        pulse_channels = [c["name"] for c in sorted(candidates, key=lambda c: c["score"], reverse=True)
                          if not c["failed_checks"]]
        out.update(
            bpm=round(stable_bpm, 1),
            quality="Good",
            pulse_channels=pulse_channels,
            signal_score=round(best["evidence_score"]),
            reason="NeuroKit pulse shapes and recent beat timing pass checks with frequency support",
        )
        return out
