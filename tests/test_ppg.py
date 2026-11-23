import numpy as np
import pytest
import neurokit2 as nk
from ppg import PPG


def signal(bpm=72, noise=.12, duration=14, fs=30):
    p = PPG()
    rng = np.random.default_rng(42)
    for t in np.arange(0, duration, 1/fs):
        p.add(t, [190, 110+4*np.sin(2*np.pi*bpm/60*t)+rng.normal(0, noise), 60])
    return p


@pytest.mark.parametrize('bpm', [45, 60, 72, 100, 140, 180])
def test_known_rates(bpm):
    result = signal(bpm).result()
    assert result['quality'] == 'Good', result
    assert abs(result['bpm']-bpm) < 4


def test_warmup():
    assert signal(duration=5).result()['bpm'] is None


def test_flat():
    p = PPG()
    for t in np.arange(0, 14, 1/30):
        p.add(t, [190, 110, 60])
    assert p.result()['bpm'] is None


def test_noise():
    p = PPG()
    rng = np.random.default_rng(4)
    for t in np.arange(0, 14, 1/30):
        p.add(t, [190, 110+rng.normal(0, 8), 60])
    assert p.result()['bpm'] is None


def test_low_fps_and_gap():
    assert signal(fs=10).result()['bpm'] is None
    p = signal()
    p.add(20, [190, 110, 60])
    assert len(p.samples) == 1
    assert p.result()['bpm'] is None


def test_clipping():
    p = signal()
    for t in np.arange(14, 16, 1/30):
        p.add(t, [255, 255, 255], 1)
    assert p.result()['bpm'] is None


def test_auto_channel_ignores_saturated_red():
    p = PPG("auto")
    rng = np.random.default_rng(12)
    for t in np.arange(0, 14, 1/30):
        green = 110 + 4*np.sin(2*np.pi*1.2*t) + rng.normal(0, .1)
        p.add(t, [255, green, 60], [1, 0, 0])
    result = p.result()
    assert result['quality'] == 'Good', result
    assert result['channel_used'] == 'green'
    assert abs(result['bpm'] - 72) < 3


def test_invalid_timestamps():
    p = signal()
    n = len(p.samples)
    p.add(float('nan'), [1, 1, 1])
    p.add(0, [1, 1, 1])
    assert len(p.samples) == n


def test_slow_brightness_drift_is_removed():
    p = PPG()
    for t in np.arange(0, 20, 1/30):
        drift = 1.4 * t
        green = 95 + drift + 4 * np.sin(2 * np.pi * 1.2 * t)
        p.add(t, [190, green, 60], [0, 0, 0], 0.02)
    result = p.result()
    assert result['quality'] == 'Good', result
    assert abs(result['bpm'] - 72) < 3


def test_second_harmonic_does_not_double_bpm():
    p = PPG()
    for t in np.arange(0, 20, 1/30):
        fundamental = 2.2 * np.sin(2 * np.pi * 1.2 * t)
        harmonic = 4.5 * np.sin(2 * np.pi * 2.4 * t)
        p.add(t, [190, 110 + fundamental + harmonic, 60], [0, 0, 0], 0.02)
    result = p.result()
    assert result['quality'] == 'Good', result
    assert abs(result['bpm'] - 72) < 4


def test_recent_phone_motion_rejects_measurement():
    p = signal(duration=14)
    for t in np.arange(14, 16, 1/30):
        green = 110 + 4 * np.sin(2 * np.pi * 1.2 * t)
        p.add(t, [190, green, 60], [0, 0, 0], 1.1)
    result = p.result()
    assert result['bpm'] is None
    assert result['quality'] == 'Poor'
    assert 'movement' in result['reason'].lower()


def test_invalid_sample_statistics_are_ignored():
    p = signal()
    count = len(p.samples)
    p.add(14, [190, 110], [0, 0, 0])
    p.add(14, [190, 110, 60], [0, 2, 0])
    p.add(14, [190, np.inf, 60], [0, 0, 0])
    assert len(p.samples) == count


@pytest.mark.parametrize('fs,bpm', [(20, 72), (30, 50), (30, 72), (30, 100),
                                    (30, 140), (30, 180), (60, 72), (60, 180)])
def test_neurokit_pulse_morphology_at_camera_rates(fs, bpm):
    """Exercise shaped pulses rather than only ideal sine waves."""
    wave = nk.ppg_simulate(duration=24, sampling_rate=fs, heart_rate=bpm,
                           frequency_modulation=0, ibi_randomness=0, random_state=42)
    p = PPG('green')
    for t, value in zip(np.arange(len(wave)) / fs, wave):
        p.add(t, [190, 100 + 8 * value, 60])
    result = p.result()
    assert result['quality'] == 'Good', result
    assert abs(result['bpm'] - bpm) < 4
    assert result['diagnostics']['template_quality'] >= 0.8
    assert 'NeuroKit2' in result['processing_method']


def test_processing_failure_returns_no_bpm(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('Injected filter failure')
    monkeypatch.setattr(nk, 'ppg_clean', fail)
    result = signal().result()
    assert result['bpm'] is None
    assert result['quality'] == 'Poor'
    assert 'processing' in result['reason']


@pytest.mark.parametrize('quality', [0.3, float('nan')])
def test_template_quality_gates_otherwise_clean_signal(monkeypatch, quality):
    monkeypatch.setattr(nk, 'ppg_quality',
                        lambda y, **kwargs: np.full(len(y), quality))
    result = signal().result()
    assert result['bpm'] is None
    assert result['quality'] == 'Poor'
    assert 'shapes' in result['reason']


def test_nyquist_guard_for_elgendi_filter():
    result = signal(fs=16).result()
    assert result['bpm'] is None
    assert '17 FPS' in result['reason']


@pytest.mark.parametrize('fs', [25, 29.7, 30, 60])
def test_adjacent_resting_rates_do_not_snap_to_69(fs):
    measured = []
    for bpm in (67, 68, 69, 70, 71):
        result = signal(bpm=bpm, noise=.06, duration=24, fs=fs).result()
        assert result['quality'] == 'Good', result
        assert abs(result['bpm'] - bpm) < .6, result
        measured.append(result['bpm'])
    assert all(b > a for a, b in zip(measured, measured[1:])), measured


def changing_signal(rates, fs=30, noise=.03):
    """Integrate instantaneous rate to maintain continuous phase at changes."""
    rng = np.random.default_rng(45)
    phase = np.cumsum(rates) / fs / 60
    p = PPG('green')
    results = []
    for i, value in enumerate(phase):
        t = i / fs
        p.add(t, [190, 110 + 4*np.sin(2*np.pi*value) + rng.normal(0, noise), 60])
        if i % (fs // 2) == 0 and t >= 10:
            results.append((t, p.result()))
    return results


@pytest.mark.parametrize('start,end', [(69, 110), (110, 69), (72, 140), (140, 72)])
def test_rate_steps_recover_without_half_rate_or_old_resting_lock(start, end):
    times = np.arange(0, 36, 1/30)
    results = changing_signal(np.where(times < 20, start, end))
    # The stability-first display intentionally adds a short median/low-pass
    # delay. Preserve an explicit eight-second settling budget, rather than
    # the previous fast-track six-second target.
    after = [(t, r) for t, r in results if t >= 28]
    assert all(r['quality'] == 'Good' for t, r in after), after
    assert all(abs(r['bpm'] - end) < 4 for t, r in after), after
    # During a transition a result may be withheld, but a corroborator's
    # subharmonic must never turn real high-rate peaks into half-rate BPM.
    assert all(min(start, end) - 4 <= r['bpm'] <= max(start, end) + 4
               for t, r in results if t >= 20 and r['bpm'] is not None)


def test_clean_rising_rate_is_not_rejected_for_variability_alone():
    times = np.arange(0, 44, 1/30)
    rates = np.clip(69 + 5*(times - 20), 69, 130)
    results = [(t, r) for t, r in changing_signal(rates) if t >= 20]
    good = [(t, r) for t, r in results if r['quality'] == 'Good']
    assert len(good) / len(results) >= .9
    for t, r in good:
        # A 5 BPM/s ramp can be ~5 s ahead of the rolling displayed rate.
        assert abs(r['bpm'] - np.clip(69 + 5*(t - 20), 69, 130)) < 26
    assert abs(good[-1][1]['bpm'] - 130) < 3


def test_fractional_cycle_lengths_do_not_break_template_epochs():
    # This smooth ramp previously intermittently raised "negative dimensions"
    # in NeuroKit's fixed-time epoch pivot, even on a noise-free waveform.
    times = np.arange(0, 24, 1/30)
    results = changing_signal(np.clip(69 + 5*(times - 20), 69, 130), noise=0)
    for t, result in results:
        assert result['quality'] == 'Good', (t, result)
        assert result['diagnostics']['template_quality'] > .9


def test_pulse_disappearance_does_not_hold_old_bpm():
    p = signal(duration=14)
    assert p.result()['quality'] == 'Good'
    for t in np.arange(14, 18, 1/30):
        p.add(t, [190, 110, 60])
    result = p.result()
    assert result['bpm'] is None
    assert result['quality'] == 'Poor'


def test_clean_transition_is_tracking_without_stale_bpm():
    times = np.arange(0, 23, 1/30)
    results = changing_signal(np.where(times < 20, 69, 110), noise=0)
    tracking = [r for t, r in results if r['quality'] == 'Tracking']
    assert tracking
    for result in tracking:
        assert result['bpm'] is None
        assert result['diagnostics']['failed_checks'] == ['disagreement']


@pytest.mark.parametrize('seed', [4, 17])
def test_noisy_constant_rate_display_is_stable_and_near_truth(seed):
    rng = np.random.default_rng(seed)
    p = PPG('green')
    measured = []
    attempted = 0
    for i, t in enumerate(np.arange(0, 55, 1/30)):
        phi = 2*np.pi*1.2*t
        value = (110 + 4*np.sin(phi) + (1+.8*np.sin(.8*t))*np.sin(2*phi+.4)
                 + .7*np.sin(.4*t) + rng.normal(0, .8))
        p.add(t, [190, value, 60])
        if i % 6 == 0 and t >= 15:
            attempted += 1
            result = p.result()
            if result['bpm'] is not None:
                measured.append(result['bpm'])
    assert len(measured) >= .8 * attempted  # Do not improve by hiding all data.
    assert np.std(measured) < .65
    assert abs(np.mean(measured) - 72) < .7
    assert max(measured) - min(measured) < 3
    assert np.max(np.abs(np.diff(measured))) < .6


def test_display_rejects_one_update_spike_but_follows_sustained_change():
    p = PPG()
    for t in np.arange(0, 4, .2):
        p._stabilize_bpm(float(t), 72)
    value = p._stabilize_bpm(4.0, 82)
    assert value == 72
    for t in np.arange(4.2, 6, .2):
        assert abs(p._stabilize_bpm(float(t), 72) - 72) < .1
    for t in np.arange(6, 12, .2):
        value = p._stabilize_bpm(float(t), 82)
    assert abs(value - 82) < .5


def test_tracker_resets_after_long_invalid_period_and_capture_gap():
    p = PPG()
    assert p._stabilize_bpm(1, 72) == 72
    assert p._stabilize_bpm(5, 110) == 110
    p.add(5, [190, 110, 60])
    p.add(8, [190, 110, 60])
    assert not p.raw_bpm_history
    assert not p.bpm_history


def channel_candidate(name, score, template=.99):
    return dict(name=name, score=score, failed_checks=[] if template >= .8 else ["template"])


def test_transient_channel_advantage_does_not_switch_measurement():
    p = PPG()
    p._select_channel([channel_candidate('green', .7)], 0)
    for t in (1, 1.2, 1.4):
        assert p._select_channel([channel_candidate('green', .5),
                                  channel_candidate('red', .9)], t)['name'] == 'green'
    assert p._select_channel([channel_candidate('green', .7),
                              channel_candidate('red', .5)], 1.6)['name'] == 'green'
    assert p.channel_switch_pending is None


def test_sustained_better_channel_switches_and_failed_channel_is_replaced():
    p = PPG()
    p._select_channel([channel_candidate('green', .7)], 0)
    options = [channel_candidate('green', .5), channel_candidate('red', .9)]
    assert p._select_channel(options, 1)['name'] == 'green'
    assert p._select_channel(options, 3.1)['name'] == 'red'
    assert p._select_channel([channel_candidate('red', .9, template=.3),
                              channel_candidate('green', .5)], 3.2)['name'] == 'green'


def faint_phone_trace(amplitude=.04, noise=.004, fs=29.9):
    p = PPG()
    rng = np.random.default_rng(17)
    for t in np.arange(0, 17, 1/fs):
        red = 200 + amplitude * np.sin(2*np.pi*80.5/60*t) + rng.normal(0, noise)
        p.add(t, [red, 95, 60], [0, 0, 0], .07)
    return p


@pytest.mark.parametrize('fs', [29.9, 60])
def test_resolved_small_phone_pulse_reports_bpm(fs):
    reading = faint_phone_trace(fs=fs).result()
    assert reading['diagnostics']['low_amplitude']
    assert reading['quality'] == 'Good', reading
    assert reading['channel_used'] == 'red'
    assert abs(reading['bpm'] - 80.5) < 1
    assert reading['signal_score'] is not None


@pytest.mark.parametrize(('amplitude', 'noise'), [(.005, 0), (.04, .035), (0, .015)])
def test_unresolved_small_variations_withhold_bpm_and_quality_score(amplitude, noise):
    reading = faint_phone_trace(amplitude, noise).result()
    assert reading['quality'] == 'Poor', reading
    assert reading['bpm'] is None and reading['signal_score'] is None
    assert reading['diagnostics']['failed_checks']


def test_faint_pulse_that_disappears_cannot_use_old_shape_agreement():
    p = faint_phone_trace()
    for t in np.arange(17, 19.2, 1/29.9):
        p.add(t, [200, 95, 60], motion=.07)
    reading = p.result()
    assert reading['quality'] == 'Poor'
    assert reading['bpm'] is None and reading['signal_score'] is None


def test_failed_recent_amplitude_gate_cannot_outvote_a_usable_color():
    p = PPG()
    p.selected_channel = 'green'
    green = dict(name='green', score=.95, failed_checks=['weak'])
    red = dict(name='red', score=.08, failed_checks=[])
    assert p._select_channel([green, red], 17)['name'] == 'red'
