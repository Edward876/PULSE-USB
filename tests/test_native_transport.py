import socket
import struct
import threading
import time

import cv2
import numpy as np
import pytest

from app import Monitor


def test_camera2_tcp_jpeg_transport():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    monitor = Monitor("phone", native_port=port)
    monitor.control(True)
    worker = threading.Thread(target=monitor.native_worker, daemon=True)
    worker.start()

    client = socket.socket()
    for _ in range(50):
        try:
            client.connect(("127.0.0.1", port))
            break
        except ConnectionRefusedError:
            time.sleep(0.01)
    else:
        raise AssertionError("Camera2 receiver did not start")

    client.sendall(b"PPG1")
    for timestamp in np.arange(0, 14, 1 / 30):
        frame = np.full((120, 160, 3), [60, 110, 190], dtype=np.uint8)
        frame[:, :, 1] = round(110 + 8 * np.sin(2 * np.pi * 1.2 * timestamp))
        ok, jpeg = cv2.imencode(".jpg", frame)
        assert ok
        payload = jpeg.tobytes()
        client.sendall(struct.pack(">QI", int(timestamp * 1_000_000_000) + 1, len(payload)) + payload)
        assert client.recv(1) == b"\x01"

    result = monitor.snapshot()
    assert result["quality"] == "Good", result
    assert abs(result["bpm"] - 72) < 3

    client.close()
    monitor.shutdown.set()
    worker.join(timeout=2)
    assert not worker.is_alive()


@pytest.mark.parametrize("protocol", [b"PPG2", b"PPG3"])
def test_camera2_tcp_raw_statistics_transport_with_verified_handshake(protocol):
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    monitor = Monitor("phone", native_port=port)
    monitor.control(True)
    worker = threading.Thread(target=monitor.native_worker, daemon=True)
    worker.start()

    client = socket.socket()
    for _ in range(50):
        try:
            client.connect(("127.0.0.1", port))
            break
        except ConnectionRefusedError:
            time.sleep(0.01)
    else:
        raise AssertionError("Camera2 receiver did not start")

    client.sendall(protocol)
    if protocol == b"PPG3":
        assert client.recv(1) == b"\x01"
    for frame_index, timestamp in enumerate(np.arange(0, 14, 1 / 30)):
        green = 110 + 7 * np.sin(2 * np.pi * 1.2 * timestamp)
        record = struct.pack(
            ">Qfffffff",
            int(timestamp * 1_000_000_000) + 1,
            220,
            green,
            55,
            0.02,
            0.0,
            0.0,
            0.03,
        )
        client.sendall(record)
        assert client.recv(1) == b"\x01"
        if frame_index == 30:
            # A duplicate timestamp is a bad frame, not a reason to tear down
            # an otherwise healthy fixed-record stream.
            client.sendall(record)
            assert client.recv(1) == b"\x01"

    result = monitor.snapshot()
    assert result["transport"] == protocol.decode("ascii")
    assert result["quality"] == "Good", result
    assert abs(result["bpm"] - 72) < 3
    assert result["diagnostics"]["motion"] < 0.1

    client.close()
    monitor.shutdown.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
