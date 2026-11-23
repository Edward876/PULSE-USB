"""Timestamp-aware phone input adapter; never pads or repeats a short recording."""
import numpy as np

BASE_MISSING_FRACTION = 0.01
SINGLE_DROP_MISSING_FRACTION = 0.03
MAX_GAP_SECONDS = 0.2501


class PhoneCaptureRejected(ValueError):
    pass


def prepare_phone_capture(rows, channel):
    channels = {"red": (1, 4), "green": (2, 5), "blue": (3, 6)}
    if channel not in channels:
        raise PhoneCaptureRejected("No pulse channel selected")
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 8 or not np.isfinite(rows).all():
        raise PhoneCaptureRejected("Invalid camera statistics")
    if len(rows) < 2 or np.any(np.diff(rows[:, 0]) <= 0):
        raise PhoneCaptureRejected("Capture timestamps must increase")
    if rows[-1, 0] - rows[0, 0] < 64:
        raise PhoneCaptureRejected("Still collecting the first 64 seconds")
    start = rows[-1, 0] - 64
    offset = max(0, np.searchsorted(rows[:, 0], start, side="right") - 1)
    recent = rows[offset:]
    dt = np.diff(recent[:, 0])
    step = float(np.median(dt))
    # Camera timestamps arrive at nanosecond precision. Allow that rounding
    # at the supported interval boundaries, then keep the model rate in range.
    if not 1 / 240 - 1e-9 <= step <= 1 / 25 + 1e-9:
        raise PhoneCaptureRejected(
            f"Stress window camera rate is {1 / step:.2f} FPS; expected 25–240 FPS"
        )
    fs = min(240, max(25, 1 / step))
    # A few missed frames are interpolated on their real timestamps. Bounds are
    # explicit engineering limits, not a claim of recovered beat timing accuracy.
    gaps = dt > 1.5 * step
    missing_seconds = float(np.sum(dt[gaps] - step))
    missing_fraction = missing_seconds / 64
    max_gap = float(dt.max())
    # Allow a larger budget only for gaps consistent with a single skipped
    # frame, with a quarter-frame margin for camera timestamp jitter.
    short_gaps_only = max_gap <= 2.25 * step
    missing_limit = SINGLE_DROP_MISSING_FRACTION if short_gaps_only else BASE_MISSING_FRACTION
    if max_gap > MAX_GAP_SECONDS or missing_fraction > missing_limit:
        raise PhoneCaptureRejected(
            f"Stress window has {missing_fraction:.1%} missing time and a longest gap of "
            f"{max_gap * 1000:.0f} ms. This window allows up to {missing_limit:.0%} missing "
            "time and a 250 ms gap."
        )
    regular = dt[~gaps]
    if not len(regular) or regular.std() / regular.mean() > .20:
        raise PhoneCaptureRejected("Capture timestamps are too irregular")
    if (np.any((recent[:, 1:4] < 0) | (recent[:, 1:4] > 255)) or
            np.any((recent[:, 4:7] < 0) | (recent[:, 4:7] > 1))):
        raise PhoneCaptureRejected("Invalid RGB or clipping data")
    vi, ci = channels[channel]
    clipping = float(recent[:, ci].mean())
    if clipping > .15:
        raise PhoneCaptureRejected(f"The {channel} channel averaged {clipping:.1%} clipped pixels; maximum is 15%")
    if recent[:, vi].mean() < 10:
        raise PhoneCaptureRejected(f"The {channel} channel is too dark; adjust finger placement")
    motion = float(np.percentile(recent[:, 7], 90))
    if motion > .65:
        raise PhoneCaptureRejected(f"Too much motion in the stress window ({motion:.2f}; maximum is 0.65)")
    count = round(64 * fs)
    actual_fs = count / 64
    grid = start + np.arange(count) / actual_fs
    raw = np.interp(grid, recent[:, 0], recent[:, vi])
    return raw, actual_fs, {"measured_fps": fs, "interpolated_seconds": missing_seconds,
                            "missing_fraction": missing_fraction, "max_capture_gap_seconds": max_gap,
                            "extended_frame_repair": missing_fraction > BASE_MISSING_FRACTION}
