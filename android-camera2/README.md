# Pulse Camera2 companion

This native Android 9+ companion enumerates Camera2 rear camera IDs and the physical camera IDs exposed by logical multi-camera devices. Some vendor firmware keeps auxiliary numeric IDs out of `getCameraIdList()` while still allowing their characteristics and direct capture; the app safely probes numeric IDs 0–7 and labels usable rear candidates. It previews the selected sensor, calculates central-region RGB/clipping values directly from YUV frames, adds accelerometer movement, and sends those raw statistics with sensor timestamps over an ADB-reversed TCP socket.

Set the Android SDK path and build:

```bash
export ANDROID_SDK_ROOT=/home/shinichi/android-sdk
bash android-camera2/build.sh
```

The build script defaults to Android Build Tools 36.0.0, which is already installed on this machine. It compiles against Android 34 and does not need Gradle or Android Studio.

Install and connect the native stream:

```bash
adb install -r android-camera2/build/pulse-camera2-debug.apk
adb reverse tcp:9001 tcp:9001
adb shell am start -n com.example.pulsecamera2/.MainActivity
```

Start the desktop server separately:

```bash
.venv/bin/python app.py --source phone --port 8001 --native-port 9001 --channel auto
```

In the Android app, allow camera permission and tap **Refresh rear lenses**. On the tested Nothing A024 firmware, the selector shows **0.6× Ultrawide — vendor camera 2**, **1× Main — camera 0**, and **Telephoto / close focus — vendor camera 3**. Select one and tap **Open selected lens**.

For the most stable PPG capture, use this order:

1. Turn on the torch.
2. Cover the selected lens gently and keep the finger still.
3. Watch **Live light**, clipping, and motion while automatic exposure settles and locks.
4. If needed, use **Lock exposure now** or **Re-adjust exposure** as a manual override.
5. Tap **Connect USB stream**, open `http://localhost:8001` on the desktop, and start measurement. Hold still for at least 10 seconds.

Choose the sensor physically closest to the flash. A sensor farther from the flash may produce a weak PPG signal even when it opens correctly. Macro is normally a focus/crop mode that reuses one of the physical sensors, not another camera module.

The PPG2 transport sends 36 bytes per accepted frame instead of a compressed image: timestamp, mean RGB, per-channel clipping fractions, and motion. Version 0.4 reads desktop acknowledgements asynchronously instead of blocking every camera frame; it disconnects only when the desktop has stopped replying for 10 seconds. One frame remains in flight at a time, so a stalled desktop does not build a stale backlog. Direct vendor-ID probing is firmware-dependent and is not guaranteed by the Camera2 API. Invalid IDs are ignored and open/configuration/network failures are shown in the app instead of crashing it. If the selector still lists only one rear sensor, that phone has rejected access to its hidden modules.
