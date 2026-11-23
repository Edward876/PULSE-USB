import numpy as np
import pytest
from stress_input import prepare_phone_capture, PhoneCaptureRejected


def rows(count=1921, step=1 / 30, origin=0):
    t = np.arange(count) * step
    data = np.zeros((count, 8))
    data[:, 0] = t + origin
    data[:, 1] = 190
    data[:, 2] = 110 + 4 * np.sin(2 * np.pi * 1.2 * t)
    data[:, 3] = 60
    return data


def test_short_infrequent_frame_drop_is_bounded_and_reported():
    x=np.delete(rows(),[800,801,802],axis=0)
    raw,fs,d=prepare_phone_capture(x,'green')
    assert len(raw)==1920 and fs==pytest.approx(30)
    assert d['missing_fraction'] < .01
    assert d['interpolated_seconds']==pytest.approx(.1)
    assert not d['extended_frame_repair']


@pytest.mark.parametrize('indices', [
    range(800, 810),
    range(100, 1800, 25),
    [*range(50, 1900, 50), 801, 802],
])
def test_large_or_frequent_missing_frames_are_rejected(indices):
    with pytest.raises(PhoneCaptureRejected):
        prepare_phone_capture(np.delete(rows(),list(indices),axis=0),'green')


def test_isolated_skipped_frames_at_two_percent_reach_model():
    # Use an exact 33 ms camera interval and bracket the full model window.
    data = rows(count=1941, step=.033, origin=10000)
    data = np.delete(data, np.arange(25, 1940, 50), axis=0)
    raw, fs, d = prepare_phone_capture(data, 'green')
    assert len(raw) == round(64 * fs)
    assert np.isfinite(raw).all()
    assert d['missing_fraction'] == pytest.approx(.02, abs=.0002)
    assert d['max_capture_gap_seconds'] == pytest.approx(.066)
    assert d['extended_frame_repair']


def test_short_recording_is_not_padded_to_64_seconds():
    with pytest.raises(PhoneCaptureRejected):
        prepare_phone_capture(rows()[:1800],'green')


def test_column_selection_uses_one_color_for_the_entire_window():
    x=rows()
    raw,fs,d=prepare_phone_capture(x,'green')
    np.testing.assert_allclose(raw,x[:-1,2],atol=1e-8)
    assert d['interpolated_seconds']==0


@pytest.mark.parametrize('fps', [25, 240])
def test_exact_camera_rate_limit_survives_timestamp_rounding(fps):
    seconds = np.arange(64 * fps + 1) / fps
    timestamps = 10000 + np.round(seconds * 1e9) / 1e9
    data = np.column_stack((timestamps, 190 + 4 * np.sin(2 * np.pi * 1.2 * seconds),
                            np.broadcast_to([110, 60, 0, 0, 0, .03], (len(seconds), 6))))
    raw, actual_fps, _ = prepare_phone_capture(data, 'red')
    assert actual_fps == fps
    assert len(raw) == 64 * fps


def test_camera_rate_below_model_minimum_still_reports_measured_rate():
    data = rows()
    data[:, 0] *= 30 / 24
    with pytest.raises(PhoneCaptureRejected, match='24.00 FPS'):
        prepare_phone_capture(data, 'green')
