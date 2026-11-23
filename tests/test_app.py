import struct
import time
import threading
import cv2
import numpy as np
from fastapi.testclient import TestClient
import app
from stress import LiveStress


def test_pages_control_and_stale():
    app.monitor = app.Monitor('phone')
    with TestClient(app.app) as c:
        assert c.get('/').status_code == 200
        assert c.get('/phone').status_code == 200
        assert c.post('/api/control', json={'running': True}, headers={'origin': 'https://foreign.example'}).status_code == 403
        assert c.post('/api/control', json={'running': True}).json()['running']
        assert c.get('/api/status').json()['bpm'] is None
        assert c.post('/api/control', json={'running': False}).json()['quality'] == 'Stopped'


def test_jpeg_usb_ingest_and_disconnect():
    app.monitor = app.Monitor('phone')
    with TestClient(app.app) as c:
        c.post('/api/control', json={'running': True})
        with c.websocket_connect('/ingest', headers={'origin': 'http://testserver'}) as ws:
            for t in np.arange(0, 14, 1/30):
                frame = np.full((120, 160, 3), [60, 110, 190], dtype=np.uint8)
                frame[:, :, 1] = round(110 + 8*np.sin(2*np.pi*1.2*t))
                ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                assert ok
                ws.send_bytes(struct.pack('<d', t*1000) + jpg.tobytes())
                assert ws.receive_json()['running']
            result = c.get('/api/status').json()
            assert result['quality'] == 'Good', result
            assert abs(result['bpm']-72) < 3
            app.monitor.last_frame = time.monotonic()-3
            waiting = c.get('/api/status').json()
            assert waiting['bpm'] is None and waiting['signal_score'] is None
            assert waiting['diagnostics'] == {}
        assert c.post('/api/control', json={'running': False}).json()['bpm'] is None


def test_capture_does_not_wait_for_slow_signal_analysis(monkeypatch):
    monitor = app.Monitor('phone')
    monitor.control(True)
    entered, release, captured = threading.Event(), threading.Event(), threading.Event()
    def slow_result(engine):
        entered.set()
        assert release.wait(3)
        return engine._empty()
    monkeypatch.setattr(app.PPG, 'result', slow_result)
    analysis = threading.Thread(target=monitor.snapshot)
    analysis.start()
    try:
        assert entered.wait(1)
        def capture():
            monitor.sample(1, [190, 110, 60], 0)
            captured.set()
        capture_thread = threading.Thread(target=capture)
        capture_thread.start()
        assert captured.wait(1), 'DSP held the capture lock'
        capture_thread.join(timeout=1)
        assert len(monitor.engine.samples) == 1
    finally:
        release.set()
        analysis.join(timeout=2)


def test_stop_during_analysis_cannot_publish_old_bpm(monkeypatch):
    monitor = app.Monitor('phone')
    monitor.control(True)
    entered, release = threading.Event(), threading.Event()
    output = []
    def slow_result(engine):
        entered.set()
        assert release.wait(3)
        return dict(engine._empty(), bpm=72, quality='Good')
    monkeypatch.setattr(app.PPG, 'result', slow_result)
    analysis = threading.Thread(target=lambda: output.append(monitor.snapshot()))
    analysis.start()
    try:
        assert entered.wait(1)
        monitor.control(False)
    finally:
        release.set()
        analysis.join(timeout=2)
    assert output[0]['bpm'] is None
    assert output[0]['quality'] == 'Stopped'


def test_capture_gap_during_analysis_cannot_publish_previous_segment(monkeypatch):
    monitor = app.Monitor('phone')
    monitor.control(True)
    monitor.sample(1, [190, 110, 60], 0)
    entered, release = threading.Event(), threading.Event()
    output = []
    def slow_result(engine):
        entered.set()
        assert release.wait(3)
        return dict(engine._empty(), bpm=72, quality='Good')
    monkeypatch.setattr(app.PPG, 'result', slow_result)
    analysis = threading.Thread(target=lambda: output.append(monitor.snapshot()))
    analysis.start()
    try:
        assert entered.wait(1)
        monitor.sample(10, [190, 110, 60], 0)
    finally:
        release.set()
        analysis.join(timeout=2)
    assert output[0]['bpm'] is None
    assert output[0]['quality'] == 'Warming up'


def test_signal_report_exports_replayable_statistics_without_video():
    app.monitor = app.Monitor('phone')
    app.monitor.control(True)
    for t in np.arange(0, 14, 1/30):
        app.monitor.sample(t, [190, 110+4*np.sin(2*np.pi*1.2*t), 60], 0)
    with TestClient(app.app) as client:
        response = client.get('/api/signal-report')
        assert response.status_code == 200
        assert 'attachment' in response.headers['content-disposition']
        report = response.json()
        assert report['pipeline_version'] == 'fingertip-quality-v4'
        data = report['signal_samples']
        assert len(data['rows']) == 420
        assert len(data['columns']) == len(data['rows'][0]) == 8
        assert data['rows'][-1][0] == 0
        assert data['rows'][0][0] < -13
        assert 'signal_samples' not in client.get('/api/status').json()


def test_live_stress_reaches_api_and_report_without_changing_bpm_buffer(monkeypatch):
    live = LiveStress(predictor=lambda rows, channel: {
        'stress_score': .25, 'threshold': .31, 'label': 'baseline_pattern',
        'signal_diagnostics': {'prv_valid': False},
    })
    app.monitor = app.Monitor('phone', stress=live)
    monkeypatch.setattr(app.PPG, 'result', lambda self: dict(self._empty(), quality='Good',
                        bpm=72, channel_used='green'))
    app.monitor.control(True)
    app.monitor.sample(0, [190, 110, 60], 0)
    app.monitor.snapshot()
    for i in range(1, 1951):
        app.monitor.sample(i/30, [190, 110+4*np.sin(2*np.pi*1.2*i/30), 60], 0)
    live.poll()
    with TestClient(app.app) as client:
        status = client.get('/api/status').json()
        assert status['bpm'] == 72
        assert status['stress']['status'] == 'Ready'
        assert status['stress']['score'] == .25
        assert 'stress_samples' not in status
        report = client.get('/api/signal-report').json()
        assert len(report['signal_samples']['rows']) <= 721
        assert len(report['stress_samples']['rows']) >= 1920
        assert report['stress_samples']['rows'][0][0] <= -64
        stopped = client.post('/api/control', json={'running': False}).json()
        assert stopped['stress']['status'] == 'Stopped'
        assert stopped['stress']['score'] is None
