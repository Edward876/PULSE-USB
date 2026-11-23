"""Research-derived PPG preparation; no classifier training in this module."""
from fractions import Fraction
from pathlib import Path
import json
import numpy as np
import neurokit2 as nk
from scipy.ndimage import median_filter
from scipy.signal import butter, detrend, resample_poly, sosfiltfilt, welch

TARGET_FS = 30
WINDOW_SECONDS = 60
GUARD_SECONDS = 2
INPUT_SECONDS = 64
PIPELINE_VERSION = "wesad-msptd-inception-v2"

class SignalRejected(ValueError):
    pass

def refine_peaks(y, peaks):
    positions = np.asarray(peaks, dtype=float).copy()
    for i, p in enumerate(peaks):
        if 0 < p < len(y) - 1:
            left, center, right = y[p - 1:p + 2]
            curvature = left - 2 * center + right
            if curvature < -1e-12:
                positions[i] += np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)
    return positions

def pulse_shape_quality(y, positions):
    cycles = []
    for start, end in zip(positions[:-1], positions[1:]):
        if not 0.30 <= (end - start) / TARGET_FS <= 1.50:
            continue
        cycle = np.interp(np.linspace(start, end, 64, endpoint=False), np.arange(len(y)), y)
        if cycle.std() > 1e-8:
            cycles.append((cycle - cycle.mean()) / cycle.std())
    if len(cycles) < 20:
        return 0.0
    cycles = np.stack(cycles)
    template = np.median(cycles, axis=0)
    template -= template.mean()
    norm = np.linalg.norm(template)
    if norm < 1e-8:
        return 0.0
    similarity = cycles @ template / (np.linalg.norm(cycles, axis=1) * norm)
    return float(np.clip(np.median(similarity), -1, 1))

def assess_peaks(clean):
    """MSPTDfast v2: no synthetic beat insertion, deletion or interval interpolation."""
    result = nk.ppg_findpeaks(clean, sampling_rate=TARGET_FS, method="msptdfast")
    peaks = np.unique(np.asarray(result["PPG_Peaks"], dtype=int))
    peaks = peaks[(peaks >= TARGET_FS) & (peaks < len(clean) - TARGET_FS)]
    if len(peaks) < 25:
        raise SignalRejected("Too few detected pulses")
    positions = refine_peaks(clean, peaks)
    ibi = np.diff(positions) / TARGET_FS
    # Local, BOUNDED tolerance. The old unbounded 3.5*MAD rule admitted gross errors.
    local = median_filter(ibi, size=9, mode="nearest")
    tolerance = np.minimum(0.20 * local, 0.150)
    valid = (ibi >= 0.30) & (ibi <= 1.50) & (np.abs(ibi - local) <= tolerance)
    fraction = float(valid.mean())
    if valid.sum() < 25:
        raise SignalRejected("Too few reliable pulse intervals")
    pair_valid = valid[:-1] & valid[1:]
    if pair_valid.sum() < 20:
        raise SignalRejected("Too few originally adjacent reliable interval pairs")
    # Require observations throughout the window, not just one clean fragment.
    midpoint_seconds = (positions[:-1] + positions[1:]) / (2 * TARGET_FS)
    block_counts = [int(np.sum(valid & (midpoint_seconds >= b) & (midpoint_seconds < b + 10)))
                    for b in range(0, WINDOW_SECONDS, 10)]
    shape = pulse_shape_quality(clean, positions)
    score = fraction + 0.25 * shape
    return {"peaks": peaks, "positions": positions, "ibi": ibi, "valid": valid,
            "pair_valid": pair_valid, "valid_fraction": fraction, "shape": shape,
            "block_counts": block_counts, "score": score}

def prepare_window(raw, sampling_rate, return_beats=False):
    """64 seconds uniform RAW PPG -> 2x1800 waveform and diagnostics.

    Literature-based components with explicit project adaptations. A quality pass
    is not proof of physiological accuracy; independent ECG audit remains necessary.
    """
    raw = np.asarray(raw, dtype=np.float64)
    fs = float(sampling_rate)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise SignalRejected("Expected a finite one-dimensional raw PPG signal")
    if not np.isfinite(fs) or not 25 <= fs <= 1000:
        raise SignalRejected("Measured acquisition rate must be at least 25 Hz")
    required = int(round(INPUT_SECONDS * fs))
    if len(raw) < required:
        raise SignalRejected(f"Need {INPUT_SECONDS} seconds of raw PPG")
    raw = raw[-required:]
    if np.std(raw) < 1e-8:
        raise SignalRejected("Flat PPG")
    ratio = Fraction(TARGET_FS / fs).limit_denominator(10000)
    uniform = resample_poly(raw, ratio.numerator, ratio.denominator)
    length = INPUT_SECONDS * TARGET_FS
    if len(uniform) < length - 1:
        raise SignalRejected("Insufficient resampled duration")
    if len(uniform) < length:
        uniform = np.pad(uniform, (0, length - len(uniform)), mode="edge")
    # Rashid et al. 2021 used a third-order 0.7–3.7 Hz Butterworth filter.
    # Our offline 64 s inference buffer uses zero-phase filtering and 2 s context.
    sos = butter(3, [0.7, 3.7], btype="bandpass", fs=TARGET_FS, output="sos")
    clean = sosfiltfilt(sos, detrend(uniform[:length]))[60:1860]
    if not np.isfinite(clean).all() or np.std(clean) < 1e-8:
        raise SignalRejected("Invalid filtered PPG")
    clean = (clean - np.mean(clean)) / np.std(clean)
    # Beat-derived descriptors are DIAGNOSTICS ONLY, never stress-model inputs.
    # Do not turn failed pulse timing into fabricated or imputed HRV measurements.
    candidates = []
    for sign in (1, -1):
        try:
            candidate = assess_peaks(sign * clean)
            candidates.append((candidate["score"], sign, candidate))
        except (SignalRejected, ValueError, IndexError, ZeroDivisionError):
            continue
    diagnostic = {"prv_valid": False, "prv_reason": "No reliable MSPTDfast beat sequence",
                  "pulse_count": None, "valid_intervals": None,
                  "rejected_interval_fraction": None, "template_correlation": None,
                  "mean_hr_bpm": None, "rmssd_ms": None, "sdnn_ms": None}
    if candidates:
        _, sign, beats = max(candidates, key=lambda value: value[0])
        clean *= sign
        failures = []
        if beats["valid_fraction"] < 0.85:
            failures.append("More than 15% locally inconsistent intervals")
        if min(beats["block_counts"]) < 4:
            failures.append("Reliable beats do not span all 10-second blocks")
        if beats["shape"] < 0.80:
            failures.append("Inconsistent pulse morphology")
        diagnostic.update(pulse_count=int(len(beats["peaks"])), valid_intervals=int(beats["valid"].sum()),
                          rejected_interval_fraction=1 - beats["valid_fraction"],
                          template_correlation=beats["shape"], prv_reason="; ".join(failures))
        if not failures:
            ibi_ms = beats["ibi"][beats["valid"]] * 1000
            changes_ms = np.diff(beats["ibi"])[beats["pair_valid"]] * 1000
            diagnostic.update(prv_valid=True, mean_hr_bpm=float(60000 / np.mean(ibi_ms)),
                              rmssd_ms=float(np.sqrt(np.mean(changes_ms ** 2))),
                              sdnn_ms=float(np.std(ibi_ms, ddof=1)))
            if return_beats:
                diagnostic["pulse_times_seconds"] = (beats["positions"] / TARGET_FS).tolist()
    # A derivative channel helps the CNN represent pulse upstroke; no ECG input.
    derivative = np.gradient(clean)
    derivative /= max(float(derivative.std()), 1e-8)
    waveform = np.clip(np.stack((clean, derivative)), -6, 6).astype(np.float32)
    return waveform, diagnostic

# Adapted single-network multi-scale residual architecture, inspired by InceptionTime.
# Not an exact reproduction of its ensemble or a claimed published stress accuracy.
import torch
from torch import nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, width=16):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, 32, 1, bias=False)
        self.paths = nn.ModuleList([nn.Conv1d(32, width, k, padding=k // 2, bias=False)
                                    for k in (9, 19, 39)])
        self.pool_path = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                       nn.Conv1d(in_channels, width, 1, bias=False))
        self.norm = nn.GroupNorm(8, 4 * width)

    def forward(self, x):
        hidden = self.bottleneck(x)
        outputs = [path(hidden) for path in self.paths]
        outputs.append(self.pool_path(x))
        return torch.relu(self.norm(torch.cat(outputs, dim=1)))

class DeterministicAdaptiveMean1d(nn.Module):
    """Adaptive-average bins using slice/mean, avoiding CUDA atomic backward."""
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        length = x.shape[-1]
        bins = []
        for i in range(self.output_size):
            start = (i * length) // self.output_size
            end = ((i + 1) * length + self.output_size - 1) // self.output_size
            bins.append(x[..., start:end].mean(dim=-1))
        return torch.stack(bins, dim=-1)

class StressNet(nn.Module):
    """One trainable model: waveform Inception blocks + learned frequency branch."""
    def __init__(self, dropout=0.4):
        super().__init__()
        self.blocks = nn.ModuleList([InceptionBlock(2 if i == 0 else 64) for i in range(6)])
        self.shortcuts = nn.ModuleList([
            nn.Sequential(nn.Conv1d(2, 64, 1, bias=False), nn.GroupNorm(8, 64)),
            nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.GroupNorm(8, 64)),
        ])
        self.frequency = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4), nn.GroupNorm(4, 16), nn.GELU(),
            nn.AvgPool1d(2), nn.Conv1d(16, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
            DeterministicAdaptiveMean1d(16), nn.Flatten(), nn.Linear(512, 64), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(192, 96), nn.GELU(), nn.Dropout(dropout), nn.Linear(96, 1))

    def forward(self, waveform):
        x, residual = waveform, waveform
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % 3 == 0:
                x = torch.relu(x + self.shortcuts[i // 3](residual))
                if i == 2:
                    x = nn.functional.avg_pool1d(x, 2)
                residual = x
        temporal = torch.cat((x.mean(dim=-1), x.std(dim=-1, unbiased=False)), dim=1)
        # FFT always float32: half-precision FFT at length 1800 is unsupported on CUDA.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            spectrum = torch.fft.rfft(waveform[:, 0].float(), dim=-1).abs().square()[:, 30:241]
            spectrum = torch.log1p(100 * spectrum / spectrum.sum(dim=-1, keepdim=True).clamp_min(1e-8))
        spectral = self.frequency(spectrum[:, None])
        return self.head(self.dropout(torch.cat((temporal, spectral), dim=1))).squeeze(1)

class StressPredictor:
    def __init__(self, bundle_dir, device="cpu"):
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(device)
        self.config = json.loads((self.bundle_dir / "model_config.json").read_text())
        if self.config["pipeline_version"] != PIPELINE_VERSION:
            raise ValueError("Incompatible preprocessing/model version")
        if self.config["input_shape"] != [2, 1800] or self.config["prv_used_as_model_input"]:
            raise ValueError("Incompatible model input contract")
        self.model = StressNet(self.config["dropout"]).to(self.device)
        self.model.load_state_dict(torch.load(self.bundle_dir / "stress_model.pt", map_location=self.device, weights_only=True))
        self.model.eval()

    def predict_prepared(self, waveform):
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.shape != (2, 1800) or not np.isfinite(waveform).all():
            raise SignalRejected("Expected a finite 2x1800 prepared waveform")
        with torch.inference_mode():
            logit = self.model(torch.as_tensor(waveform[None], device=self.device))
            score = float(torch.sigmoid(logit).cpu().item())
        return {"stress_score": score, "threshold": self.config["threshold"],
                "label": "elevated_stress_pattern" if score >= self.config["threshold"] else "baseline_pattern",
                "experimental": True, "phone_validation": "not established",
                "interpretation": "Uncalibrated WESAD class score; not stress severity or a diagnosis"}

    def predict_raw(self, raw, sampling_rate):
        wave, diagnostic = prepare_window(raw, sampling_rate)
        return {**self.predict_prepared(wave), "signal_diagnostics": diagnostic,
                "signal_caution": "Quality gates cannot establish that a periodic signal is physiological"}

    def predict_phone_rows(self, rows, channel):
        channels = {"red": (1, 4), "green": (2, 5), "blue": (3, 6)}
        if channel not in channels:
            raise SignalRejected("Keep one red/green/blue channel fixed for the recording")
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 8 or not np.isfinite(rows).all():
            raise SignalRejected("Expected finite time/RGB/clipping/motion rows with 8 columns")
        if len(rows) < 2 or np.any(np.diff(rows[:, 0]) <= 0):
            raise SignalRejected("Capture timestamps must strictly increase")
        end = rows[-1, 0]
        if end - rows[0, 0] < INPUT_SECONDS:
            raise SignalRejected("Need 64 seconds of real raw samples")
        start = end - INPUT_SECONDS
        offset = max(0, np.searchsorted(rows[:, 0], start, side="right") - 1)
        recent = rows[offset:]
        dt = np.diff(recent[:, 0])
        fs = 1 / np.median(dt)
        if fs < 25 or dt.max() > .10 or dt.std() / dt.mean() > .20:
            raise SignalRejected("Low/irregular FPS or frame gaps")
        vi, ci = channels[channel]
        if (np.any((recent[:, 1:4] < 0) | (recent[:, 1:4] > 255)) or
            np.any((recent[:, 4:7] < 0) | (recent[:, 4:7] > 1))):
            raise SignalRejected("Invalid RGB or clipping data")
        if recent[:, ci].mean() > .15 or recent[:, vi].mean() < 10:
            raise SignalRejected("Clipped or dark recording")
        if np.percentile(recent[:, 7], 90) > .65:
            raise SignalRejected("Excessive phone motion")
        count = round(INPUT_SECONDS * fs)
        actual_fs = count / INPUT_SECONDS
        raw = np.interp(start + np.arange(count) / actual_fs, recent[:, 0], recent[:, vi])
        return self.predict_raw(raw, actual_fs)
