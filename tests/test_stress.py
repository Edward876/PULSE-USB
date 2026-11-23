import json
import threading
from types import SimpleNamespace

import numpy as np

from stress import LiveStress, ModelProcess, read_config
from stress_input import PhoneCaptureRejected, prepare_phone_capture


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def result(score=.2):
    return {"stress_score": score, "threshold": .31,
            "label": "baseline_pattern" if score < .31 else "elevated_stress_pattern",
            "signal_diagnostics": {"prv_valid": False}}


def make(predictor=None):
    clock = Clock()
    calls = []
    def predict(rows, channel):
        calls.append((rows, channel))
        return result()
    live = LiveStress(predictor=predictor or predict, clock=clock)
    live.control(True)
    live.observe_quality({"quality": "Good", "channel_used": "green"})
    return live, clock, calls


def feed(live, clock, first, last):
    for index in range(first, last + 1):
        t = index / 30
        clock.now = 1000 + t
        if index % 15 == 0:
            live.observe_quality({"quality": "Good", "channel_used": "green"})
        live.add((t, 190, 110 + 4 * np.sin(2 * np.pi * 1.2 * t), 60, 0, 0, 0, .03))


def test_first_full_window_and_30_second_updates_use_real_bracketing_samples():
    live, clock, calls = make()
    feed(live, clock, 0, 1919)
    live.poll()
    assert not calls
    assert live.snapshot()["status"] == "Collecting"
    feed(live, clock, 1920, 1920)
    live.poll()
    assert len(calls) == 1
    assert calls[0][0][-1][0] - calls[0][0][0][0] >= 64
    assert live.snapshot()["status"] == "Ready"
    assert live.snapshot()["score"] == .2
    feed(live, clock, 1921, 2819)
    live.poll()
    assert len(calls) == 1
    feed(live, clock, 2820, 2820)
    live.poll()
    assert len(calls) == 2
    assert len(live.rows) <= 1922
    json.dumps(live.snapshot(), allow_nan=False)


def test_stop_clears_score_and_restarts_warmup():
    live, clock, _ = make()
    feed(live, clock, 0, 1920)
    live.poll()
    live.control(False)
    assert live.snapshot()["status"] == "Stopped"
    assert live.snapshot()["score"] is None
    assert live.snapshot()["previous_estimate"] is None
    live.control(True)
    assert live.snapshot()["collected_seconds"] == 0


def test_restarted_session_tries_another_pulse_channel_on_capture_rejection():
    live, clock, _ = make()
    feed(live, clock, 0, 1920)
    live.poll()
    assert live.snapshot()["score"] == .2
    live.control(False)
    live.control(True)
    assert live.snapshot()["score"] is None
    attempts = []

    def predict(rows, channel):
        attempts.append(channel)
        try:
            prepare_phone_capture(rows, channel)
        except PhoneCaptureRejected as error:
            return {"type": "rejected", "reason": str(error)}
        return {"type": "result", "result": result(.8)}

    live.predictor = None
    live.config = {}
    live.process = SimpleNamespace(predict=predict)
    quality = {"quality": "Good", "channel_used": "red", "pulse_channels": ["red", "green"]}
    for index in range(1921):
        t = index / 30
        clock.now = 1070 + t
        live.observe_quality(quality)
        pulse = np.sin(2 * np.pi * 1.2 * t)
        live.add((1070 + t, 190 + 4 * pulse, 110 + 2 * pulse, 60, .17, 0, 0, .03))
        if index == 1919:
            live.poll()
            assert not attempts

    live.poll()
    reading = live.snapshot()
    assert attempts == ["red", "green"]
    assert reading["status"] == "Ready"
    assert reading["score"] == .8
    assert reading["channel"] == "green"


def test_large_camera_gap_marks_previous_result_stale_and_requires_new_window():
    live, clock, calls = make()
    feed(live, clock, 0, 1920)
    live.poll()
    feed(live, clock, 1930, 1930)
    assert live.snapshot()["status"] == "Stale"
    assert live.snapshot()["score"] is None
    assert live.snapshot()["previous_estimate"]["score"] == .2
    assert live.snapshot()["collected_seconds"] == 0
    live.poll()
    assert len(calls) == 1


def test_offline_wall_clock_invalidates_live_value_even_without_new_frames():
    live, clock, _ = make()
    feed(live, clock, 0, 1920)
    live.poll()
    clock.now += 3
    assert live.snapshot()["status"] == "Waiting"
    assert live.snapshot()["score"] is None


def test_only_sustained_contact_loss_resets_capture_not_brief_quality_changes():
    live, clock, _ = make()
    feed(live, clock, 0, 300)
    live.add((10.03, 190, 110, 60, 0, .5, 0, 0))
    assert live.rows and live.channel == 'green'
    live.observe_quality({"quality": "Poor"})
    clock.now += 2.1
    live.observe_quality({"quality": "Poor"})
    assert live.rows and live.channel == 'green'
    for i in range(301, 321):
        live.add((i/30, 190, 0, 60, 0, 1, 0, 0))
    assert live.channel is None
    assert live.snapshot()['collected_seconds'] <= .2
    assert live.snapshot()['status'] == 'Acquiring pulse'


def test_tracking_hides_old_score_without_destroying_physical_capture():
    live, clock, _ = make()
    feed(live, clock, 0, 1920)
    live.poll()
    live.observe_quality({"quality": "Tracking"})
    assert live.snapshot()["score"] is None
    assert live.snapshot()["status"] == "Stale"
    assert len(live.rows) >= 1920


def test_channel_preference_change_keeps_all_rgb_history_for_fixed_column_inference():
    live, clock, _ = make()
    feed(live, clock, 0, 1920)
    live.poll()
    live.observe_quality({'quality': 'Good', 'channel_used': 'red'})
    assert live.channel == 'red'
    assert len(live.rows) >= 1920
    assert live.snapshot()['score'] == .2
    assert live.last_estimate['channel'] == 'green'
    assert live.snapshot()['reset_count'] == 0


def test_slow_inference_does_not_block_capture_and_stop_discards_its_result():
    entered, release = threading.Event(), threading.Event()
    def slow(rows, channel):
        entered.set()
        assert release.wait(3)
        return result(.9)
    live, clock, _ = make(slow)
    feed(live, clock, 0, 1920)
    worker = threading.Thread(target=live.poll)
    worker.start()
    try:
        assert entered.wait(1)
        captured = threading.Event()
        def add():
            feed(live, clock, 1921, 1921)
            captured.set()
        capture = threading.Thread(target=add)
        capture.start()
        assert captured.wait(1)
        live.control(False)
        live.control(True)
    finally:
        release.set()
        worker.join(2)
    assert live.last_estimate is None


def test_demo_never_classifies_synthetic_bpm_as_stress():
    live, clock, calls = make()
    live.source = "demo"
    feed(live, clock, 0, 2000)
    live.poll()
    assert not calls
    assert live.snapshot()["status"] == "Demo"


def test_missing_environment_is_reported_without_crashing_monitor():
    live = LiveStress({"python": "/missing/python", "bundle_dir": "/missing/bundle"})
    live.start_worker()
    live.thread.join(2)
    assert live.snapshot()["status"] == "Unavailable"
    assert live.snapshot()["score"] is None
    live.close()


def test_model_rejection_retains_window_retries_and_does_not_publish_old_score_as_current():
    live, clock, _ = make()
    feed(live, clock, 0, 1920)
    live.poll()
    live.predictor = None
    live.config = {}
    live.process = SimpleNamespace(predict=lambda rows, channel: {"type": "rejected", "reason": "Irregular frame timing"})
    feed(live, clock, 1921, 2820)
    live.poll()
    assert live.snapshot()["score"] is None
    assert live.snapshot()["status"] == "Stale"
    assert live.snapshot()["collected_seconds"] == 64
    assert live.snapshot()['next_update_seconds'] == 5
    live.predictor = lambda rows, channel: result(.4)
    feed(live, clock, 2821, 2970)
    live.poll()
    assert live.snapshot()['score'] == .4
    assert live.snapshot()['status'] == 'Ready'


def test_pinned_bundle_configuration_points_to_successfully_trained_export():
    config = read_config()
    assert config["bundle_dir"].is_dir()
    assert (config["bundle_dir"] / "stress_model.pt").is_file()


def test_invalid_model_score_invalidates_previous_live_result():
    live, clock, _ = make()
    feed(live, clock, 0, 1920)
    live.poll()
    live.predictor = lambda rows, channel: result(float('nan'))
    feed(live, clock, 1921, 2820)
    live.poll()
    assert live.snapshot()['score'] is None
    assert live.snapshot()['status'] == 'Unavailable'


def test_brief_timestamp_gap_and_delayed_delivery_do_not_discard_the_minute():
    live, clock, _ = make()
    feed(live, clock, 0, 900)
    clock.now += .8
    live.add((901/30, 190, 110, 60, 0, 0, 0, .03))
    assert len(live.rows) == 902
    live.add((905/30, 190, 110, 60, 0, 0, 0, .03))
    assert len(live.rows) == 903
    assert live.snapshot()['reset_count'] == 0


def test_repeated_rgb_changes_and_quality_blips_still_produce_first_result():
    live, clock, calls = make()
    for i in range(1921):
        clock.now = 1000 + i/30
        if i%30 == 0:
            live.observe_quality({'quality':'Good','channel_used':'red' if i%60 else 'green'})
        elif i%30 == 10:
            live.observe_quality({'quality':'Poor'})
        live.add((i/30,190,110+4*np.sin(2*np.pi*1.2*i/30),60,0,0,0,.03))
    live.observe_quality({'quality':'Good','channel_used':'green'})
    live.poll()
    assert len(calls) == 1
    assert live.snapshot()['score'] == .2
    assert live.snapshot()['reset_count'] == 0


def test_long_unconfirmed_capture_only_buffers_the_recent_analysis_window():
    live, clock, calls = make()
    live.control(True)
    for i in range(1981):
        clock.now = 1000 + i/30
        live.add((i/30, 190, 110, 60, 0, 0, 0, .07))
    live.poll()
    state = live.snapshot()
    assert state['status'] == 'Acquiring pulse' and state['collected_seconds'] == 8
    assert not calls
    live.observe_quality({'quality':'Good', 'channel_used':'red'})
    live.poll()
    assert not calls
    feed(live, clock, 1981, 3660)
    live.poll()
    assert len(calls) == 1
    assert calls[0][0][-1][0] - calls[0][0][0][0] == 64


def test_acquisition_only_credits_contiguous_exposed_rows_of_confirmed_color():
    live, clock, _ = make()
    live.control(True)
    for i in range(301):
        clock.now = 1000 + i/30
        live.add((i/30, 190 if i>=271 else 0, 110, 60, 0, 0, 0, .03))
    assert live.snapshot()['collected_seconds'] == 8
    live.observe_quality({'quality':'Good', 'channel_used':'red'})
    assert live.rows[0][0] == 271/30
    assert live.snapshot()['collected_seconds'] < 1
