# Pulse USB PPG

Real-time fingertip photoplethysmography on Linux using an Android phone as a USB-connected camera. The system streams camera statistics to a local Python service, extracts a PPG waveform, estimates beats per minute (BPM), checks signal quality, and displays the result in a browser.

An optional experimental classifier can process a longer clean PPG window and return a WESAD baseline-versus-stress pattern score. It was trained on wrist BVP data, not phone recordings.

> **Research and education only.** This project is not a medical device, diagnostic tool, or clinical HRV/stress assessment system.

## Features

- Native Android Camera2 companion with rear-camera selection and torch control
- USB transport through ADB port reversal
- Live PPG waveform, BPM, frame rate, and capture diagnostics
- Automatic red, green, and blue channel selection
- NeuroKit2 Elgendi PPG cleaning and peak detection
- FFT and autocorrelation checks before accepting BPM
- Optional PRV diagnostics when beat timing passes quality checks
- Optional WESAD-trained waveform classifier
- Local processing with no cloud service or API key

## System overview

```text
Android camera + flash
        │
        ▼
Central ROI statistics
RGB + clipping + motion + timestamp
        │
        ▼
ADB USB transport
        │
        ▼
Python PPG service
quality checks → PPG cleaning → pulse detection → BPM
        │
        ▼
Local web dashboard
```

The native companion measures the central 60% of each YUV frame and sends compact statistics instead of full video. The desktop service performs the PPG processing.

## Repository layout

```text
app.py                 FastAPI server, phone receiver, camera capture
ppg.py                 Real-time PPG, BPM, and signal-quality pipeline
stress.py              Rolling stress-window controller
stress_input.py        Long-window timing and capture-quality checks
stress_worker.py       Separate model inference process
stress_model.json      Pinned model bundle and model-Python path
static/                Local web interface
android-camera2/       Android Camera2 companion
stress_model_training.ipynb
                       WESAD preprocessing, model training, and export
wesad_dataset_eda.ipynb
                       Dataset exploration notebook
tests/                 Automated regression tests
artifacts/             Saved model bundles and training outputs
```

## Requirements

- Linux desktop
- Python 3.11 or newer
- Android phone with USB debugging enabled
- Data-capable USB cable
- ADB

On Arch Linux:

```bash
sudo pacman -Syu python python-pip android-tools
```

Create the desktop environment:

```bash
cd ~/code/pulse-usb
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Quick start: desktop demo

Run the simulated source first to confirm the dashboard and estimator work:

```bash
cd ~/code/pulse-usb
.venv/bin/python app.py --source demo
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), then select **Start measurement**. Demo mode generates synthetic data and does not produce a real stress result.

## Recommended setup: native Android Camera2 over USB

### 1. Build and install the Android companion

Set the Android SDK location, build the APK, and install it:

```bash
cd ~/code/pulse-usb
export ANDROID_SDK_ROOT=/home/shinichi/android-sdk
bash android-camera2/build.sh
adb install -r android-camera2/build/pulse-camera2-debug.apk
```

### 2. Start the desktop server

```bash
cd ~/code/pulse-usb
.venv/bin/python app.py --source phone --port 8001 --native-port 9001 --channel auto
```

### 3. Connect and open the phone application

In another terminal:

```bash
adb devices -l
adb reverse tcp:9001 tcp:9001
adb shell am start -n com.example.pulsecamera2/.MainActivity
xdg-open http://127.0.0.1:8001
```

The phone must appear as `device` in `adb devices -l`. In the Android application:

1. Allow camera permission.
2. Refresh rear lenses.
3. Choose the rear lens nearest to the flash.
4. Open the lens and enable the torch.
5. Cover the selected lens gently with a fingertip.
6. Wait for exposure to settle, then connect the USB stream.
7. Select **Start measurement** in the desktop page.

For device-specific camera details, see [android-camera2/README.md](android-camera2/README.md).

## Live PPG pipeline

The PPG estimator:

1. Validates timestamps and rejects duplicate, backward, or malformed samples.
2. Rejects channels that are too dark or excessively clipped.
3. Selects a stable RGB channel. Automatic mode avoids switching unless another channel remains substantially better.
4. Interpolates camera samples onto a uniform time grid.
5. Removes isolated spikes with a median filter.
6. Cleans the signal with NeuroKit2 using the Elgendi method.
7. Estimates BPM from pulse peaks, FFT frequency, and autocorrelation.
8. Shows BPM only when the independent estimates agree.

| Setting | Value |
| --- | ---: |
| BPM buffer | 24 seconds |
| BPM warm-up | 10 seconds |
| Recent rate-analysis window | 8 seconds |
| Minimum accepted frame rate | 17 fps |
| Channel brightness floor | 10 / 255 |
| Clipping rejection threshold | 30% |
| Capture gap that resets BPM | 0.5 seconds |

The signal-quality score uses brightness, clipping, motion, timing regularity, spectral concentration, peak consistency, autocorrelation, and pulse-template similarity.

## Stress-pattern model

The optional model is an experimental binary classifier trained with WESAD wrist BVP:

| Item | Value |
| --- | --- |
| Source data | WESAD wrist BVP |
| Participants in local copy | 15 |
| Prepared windows | 5,150 |
| Model input | 60 seconds at 30 Hz |
| Input shape | 2 × 1,800 |
| Input channels | Normalized PPG and its derivative |
| Architecture | Six-block residual Inception CNN with frequency branch |
| Trainable parameters | 280,833 |
| Final threshold | 0.31 |

The live classifier needs 64 seconds of usable data because it reserves filter context before processing the central 60-second input. It refreshes the current estimate every 30 seconds.

The worker rejects long gaps, unstable timing, severe clipping, low frame rate, and excessive motion. It may repair tightly bounded short gaps, but it does not pad or repeat a short recording to create a 64-second input.

### Stress score display

| Score | Display label |
| --- | --- |
| Below 0.10 | Strongly baseline-like |
| 0.10 to below 0.26 | Baseline-leaning |
| 0.26 to 0.36 | Borderline |
| Above 0.36 to below 0.80 | Stress-leaning |
| 0.80 and above | Strongly stress-like |

These labels describe an uncalibrated WESAD class score. They do not describe a person's stress severity.

### Run without the model

```bash
.venv/bin/python app.py --source phone --port 8001 --native-port 9001 --no-stress
```

## WESAD preprocessing and training

The active training notebook is [stress_model_training.ipynb](stress_model_training.ipynb). Run it in a Python 3.11 or 3.12 Miniconda environment with PyTorch.

The preprocessing procedure:

1. Keep continuous WESAD baseline and stress segments.
2. Extract 64-second wrist-BVP contexts every 5 seconds.
3. Resample from 64 Hz to 30 Hz.
4. Detrend and apply a 0.7 to 3.7 Hz Butterworth band-pass filter.
5. Remove 2 seconds from both ends.
6. Retain the central 60 seconds, or 1,800 samples.
7. Normalize the PPG waveform and its first derivative.

Development uses grouped participant folds. Windows from the same participant do not appear in both training and validation within a fold.

The saved model bundle is pinned by [stress_model.json](stress_model.json). Keep the selected bundle together.

## Alternative capture paths

### Browser USB bridge

```bash
cd ~/code/pulse-usb
.venv/bin/python app.py --source phone
```

In another terminal:

```bash
adb reverse tcp:8000 tcp:8000
adb shell am start -a android.intent.action.VIEW -d http://localhost:8000/phone
```

Use the phone page to select a camera and start the stream. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) on the desktop.

### V4L2 or scrcpy camera bridge

If Android exposes a camera as a V4L2 device:

```bash
.venv/bin/python app.py --source camera --device /dev/video10 --channel auto
```

The desktop reads the central 60% of the frame. A V4L2 source opening successfully does not guarantee that fingertip illumination is adequate for PPG.

## Tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

The tests cover PPG timing, quality rejection, transport protocols, web API behavior, rolling-window handling, and model-input validation. They verify software behavior; they do not validate human BPM, HRV, or stress accuracy.

## Limitations

- Phone-camera PPG accuracy has not been compared against a synchronized clinical reference.
- PRV from PPG is not interchangeable with ECG-derived HRV.
- WESAD uses wrist BVP, while this application measures fingertip camera PPG.
- The stress classifier has not been validated on labelled phone recordings.
- Motion, poor contact, exposure variation, and periodic artefacts can produce misleading PPG signals.

## References

- S. Lee, C. G. Park, and K. W. Ko, [Photoplethysmography Measurement Algorithm for a Smartphone Camera](https://doi.org/10.1587/transcom.2015MIP0001), 2016.
- P. Schmidt et al., [Introducing WESAD](https://doi.org/10.1145/3242969.3242985), 2018.
- D. Makowski et al., [NeuroKit2](https://doi.org/10.3758/s13428-020-01516-y), 2021.
- P. H. Charlton et al., [MSPTDfast](https://doi.org/10.1088/1361-6579/adb89e), 2025.
