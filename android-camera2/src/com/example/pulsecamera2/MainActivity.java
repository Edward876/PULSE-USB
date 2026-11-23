package com.example.pulsecamera2;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity implements SensorEventListener {
    private static final int CAMERA_PERMISSION = 41;
    private static final int DEFAULT_NATIVE_PORT = 9001;
    private static final long DESKTOP_ACK_TIMEOUT_NS = 10_000_000_000L;
    private static final String TAG = "PulseCamera2";

    private CameraManager cameraManager;
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private HandlerThread cameraThread;
    private Handler cameraHandler;
    private Executor cameraExecutor;
    private final ExecutorService networkExecutor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean frameInFlight = new AtomicBoolean(false);

    private TextureView preview;
    private Spinner cameraSpinner;
    private EditText portField;
    private Button openButton;
    private Button connectButton;
    private Button torchButton;
    private Button exposureButton;
    private Button stopButton;
    private TextView status;
    private TextView signalGuide;

    private final List<CameraChoice> choices = new ArrayList<>();
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest.Builder requestBuilder;
    private ImageReader imageReader;
    private Surface previewSurface;
    private CameraChoice activeChoice;
    private boolean torchEnabled;
    private boolean exposureLocked;
    private boolean aeLockSupported;
    private boolean awbLockSupported;
    private Range<Integer> exposureCompensationRange;
    private int exposureCompensation;
    private int exposureStableFrames;
    private int exposureFrameCounter;
    private long lastGuideUpdateNs;
    private volatile float motionLevel;
    private final float[] lastAcceleration = new float[3];
    private boolean haveAcceleration;

    private volatile Socket socket;
    private volatile DataInputStream socketInput;
    private volatile DataOutputStream socketOutput;
    private volatile boolean connected;
    private volatile long lastDesktopAckNs;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager == null ? null
                : sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        cameraThread = new HandlerThread("PulseCamera2");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());
        cameraExecutor = command -> cameraHandler.post(command);
        buildInterface();
    }

    private void buildInterface() {
        int pad = dp(20);
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        scroll.addView(root, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView title = text("Pulse Camera2", 30);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        root.addView(title);
        root.addView(text("Choose a rear sensor, verify its preview, then stream it over USB.", 16));

        Button refreshButton = button("Refresh rear lenses");
        refreshButton.setOnClickListener(v -> requestOrEnumerate());
        root.addView(refreshButton, margins());

        cameraSpinner = new Spinner(this);
        cameraSpinner.setEnabled(false);
        root.addView(cameraSpinner, margins());

        openButton = button("Open selected lens");
        openButton.setEnabled(false);
        openButton.setOnClickListener(v -> openSelectedCamera());
        root.addView(openButton, margins());

        preview = new TextureView(this);
        LinearLayout.LayoutParams previewParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, dp(360));
        previewParams.setMargins(0, dp(10), 0, dp(10));
        root.addView(preview, previewParams);

        signalGuide = text("Live capture: open a lens to check light and motion.", 14);
        signalGuide.setPadding(0, dp(4), 0, dp(10));
        root.addView(signalGuide);

        TextView portLabel = text("Desktop Camera2 port", 14);
        root.addView(portLabel);
        portField = new EditText(this);
        portField.setInputType(android.text.InputType.TYPE_CLASS_NUMBER);
        portField.setText(String.valueOf(DEFAULT_NATIVE_PORT));
        root.addView(portField, margins());

        connectButton = button("Connect USB stream");
        connectButton.setEnabled(false);
        connectButton.setOnClickListener(v -> {
            if (connected) disconnectNetwork("USB stream disconnected.");
            else connectNetwork();
        });
        root.addView(connectButton, margins());

        torchButton = button("Torch on");
        torchButton.setEnabled(false);
        torchButton.setOnClickListener(v -> toggleTorch());
        root.addView(torchButton, margins());

        exposureButton = button("Lock exposure now");
        exposureButton.setEnabled(false);
        exposureButton.setOnClickListener(v -> toggleExposureLock());
        root.addView(exposureButton, margins());

        stopButton = button("Stop camera");
        stopButton.setEnabled(false);
        stopButton.setOnClickListener(v -> {
            disconnectNetwork("USB stream disconnected.");
            closeCamera();
            setStatus("Camera stopped. Select a lens to start again.");
        });
        root.addView(stopButton, margins());

        status = text("Tap Refresh rear lenses and allow camera permission.", 15);
        status.setPadding(0, dp(14), 0, dp(14));
        root.addView(status);
        root.addView(text("With the torch on, exposure adjusts automatically and locks when the fingertip lighting is stable. Use the exposure button to override it.", 13));
        root.addView(text("Macro is often a focus mode that reuses the ultrawide or telephoto sensor, so it may not appear as a separate camera.", 13));
        root.addView(text("Camera2 normally shows only public sensors. On compatible firmware this app can also open vendor numeric camera IDs.", 13));
        root.addView(text("Educational prototype only — not a medical device or diagnostic tool.", 13));
        setContentView(scroll);
    }

    private LinearLayout.LayoutParams margins() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.setMargins(0, dp(8), 0, dp(4));
        return params;
    }

    private TextView text(String value, int size) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(0xff0a1c43);
        return view;
    }

    private Button button(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        return button;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void requestOrEnumerate() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
        } else {
            enumerateCameras();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == CAMERA_PERMISSION && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) {
            enumerateCameras();
        } else if (requestCode == CAMERA_PERMISSION) {
            setStatus("Camera permission is required to list and open lenses.");
        }
    }

    private void enumerateCameras() {
        closeCamera();
        choices.clear();
        int physicalCount = 0;
        int vendorCandidateCount = 0;
        try {
            String[] publicIds = cameraManager.getCameraIdList();
            Log.i(TAG, "CameraManager public IDs: " + Arrays.toString(publicIds));
            Set<String> publicIdSet = new HashSet<>(Arrays.asList(publicIds));
            for (int candidate = 0; candidate <= 7; candidate++) {
                String candidateId = String.valueOf(candidate);
                if (publicIdSet.contains(candidateId)) continue;
                try {
                    CameraCharacteristics hidden = cameraManager.getCameraCharacteristics(candidateId);
                    Log.i(TAG, "Firmware permits characteristics for non-public camera "
                            + candidateId + lensSummaryForLog(hidden));
                    Integer facing = hidden.get(CameraCharacteristics.LENS_FACING);
                    float[] focalLengths = hidden.get(
                            CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
                    float focal = focalLengths == null || focalLengths.length == 0
                            ? Float.NaN : focalLengths[0];
                    if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK
                            && Float.isFinite(focal)
                            && !hasCapability(hidden,
                            CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)) {
                        choices.add(cameraChoice(candidateId, null, hidden, true));
                        vendorCandidateCount++;
                    }
                } catch (IllegalArgumentException | CameraAccessException | SecurityException exception) {
                    Log.i(TAG, "Non-public camera " + candidateId + " rejected: "
                            + exception.getClass().getSimpleName());
                }
            }
            for (String openId : publicIds) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(openId);
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) continue;

                choices.add(cameraChoice(openId, null, characteristics, false));
                Set<String> seen = new HashSet<>();
                for (String physicalId : characteristics.getPhysicalCameraIds()) {
                    if (!seen.add(physicalId)) continue;
                    CameraCharacteristics physical = safeCharacteristics(physicalId, characteristics);
                    choices.add(cameraChoice(openId, physicalId, physical, false));
                    physicalCount++;
                }
            }
        } catch (CameraAccessException | SecurityException exception) {
            setStatus("Could not enumerate Camera2 devices: " + exception.getMessage());
            return;
        }

        float mainFocalLength = Float.NaN;
        for (CameraChoice choice : choices) {
            if (choice.physicalId == null && !choice.vendorDirect
                    && Float.isFinite(choice.focalLength)) {
                mainFocalLength = choice.focalLength;
                break;
            }
        }
        final float focalReference = mainFocalLength;
        choices.sort(Comparator.comparingDouble(choice ->
                Float.isFinite(choice.focalLength) ? choice.focalLength : Double.MAX_VALUE));
        List<String> labels = new ArrayList<>();
        for (CameraChoice choice : choices) {
            choice.label = friendlyLensLabel(choice, focalReference);
            labels.add(choice.label);
            Log.i(TAG, "Selectable rear sensor: " + choice.label);
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, labels);
        cameraSpinner.setAdapter(adapter);
        cameraSpinner.setEnabled(!choices.isEmpty());
        openButton.setEnabled(!choices.isEmpty());
        if (choices.isEmpty()) {
            setStatus("No rear Camera2 devices were exposed to this app.");
        } else {
            setStatus("Found " + choices.size() + " rear sensor choices ("
                    + physicalCount + " standard auxiliary, " + vendorCandidateCount
                    + " vendor-hidden). Tap the selector to choose 0.6×, 1×, or telephoto.");
        }
    }

    private CameraCharacteristics safeCharacteristics(String id, CameraCharacteristics fallback) {
        try {
            return cameraManager.getCameraCharacteristics(id);
        } catch (Exception ignored) {
            return fallback;
        }
    }

    private CameraChoice cameraChoice(String openId, String physicalId,
                                      CameraCharacteristics characteristics, boolean vendorDirect) {
        float[] focalLengths = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
        Boolean flash = characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
        Float minimumFocus = characteristics.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE);
        float focal = focalLengths == null || focalLengths.length == 0 ? Float.NaN : focalLengths[0];
        return new CameraChoice(openId, physicalId, focal, Boolean.TRUE.equals(flash),
                minimumFocus == null ? 0f : minimumFocus, vendorDirect);
    }

    private String lensSummaryForLog(CameraCharacteristics characteristics) {
        float[] values = characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
        return values == null || values.length == 0
                ? ""
                : String.format(Locale.US, " (%.2f mm)", values[0]);
    }

    private boolean hasCapability(CameraCharacteristics characteristics, int wanted) {
        int[] capabilities = characteristics.get(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
        if (capabilities == null) return false;
        for (int capability : capabilities) {
            if (capability == wanted) return true;
        }
        return false;
    }

    private String friendlyLensLabel(CameraChoice choice, float mainFocalLength) {
        String name;
        if (Float.isFinite(mainFocalLength) && choice.focalLength < mainFocalLength * 0.75f) {
            name = "0.6× Ultrawide";
        } else if (Float.isFinite(mainFocalLength) && choice.focalLength > mainFocalLength * 1.25f) {
            name = choice.minimumFocusDistance > 0
                    ? "Telephoto / close focus (3× or Macro mode)"
                    : "Telephoto";
        } else if (choice.physicalId == null && !choice.vendorDirect) {
            name = "1× Main";
        } else {
            name = "Auxiliary rear lens";
        }
        String id = choice.vendorDirect
                ? "vendor camera " + choice.openId
                : choice.physicalId == null
                ? "camera " + choice.openId
                : "physical " + choice.physicalId + " via camera " + choice.openId;
        String focal = Float.isFinite(choice.focalLength)
                ? String.format(Locale.US, "%.2f mm", choice.focalLength)
                : "unknown focal length";
        return name + " — " + id + " — " + focal + (choice.flashAvailable ? " — flash" : "");
    }

    private void openSelectedCamera() {
        if (choices.isEmpty()) return;
        if (!preview.isAvailable()) {
            preview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                    preview.setSurfaceTextureListener(null);
                    openSelectedCamera();
                }
                @Override public void onSurfaceTextureSizeChanged(SurfaceTexture s, int w, int h) {}
                @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture s) { return true; }
                @Override public void onSurfaceTextureUpdated(SurfaceTexture s) {}
            });
            setStatus("Waiting for preview surface…");
            return;
        }
        closeCamera();
        activeChoice = choices.get(cameraSpinner.getSelectedItemPosition());
        CameraCharacteristics streamCharacteristics = activeChoice.physicalId == null
                ? safeCharacteristics(activeChoice.openId, null)
                : safeCharacteristics(activeChoice.physicalId,
                    safeCharacteristics(activeChoice.openId, null));
        Size size = chooseCaptureSize(streamCharacteristics);
        if (size == null) {
            setStatus("The selected lens exposes no YUV stream size.");
            return;
        }
        imageReader = ImageReader.newInstance(size.getWidth(), size.getHeight(),
                ImageFormat.YUV_420_888, 2);
        imageReader.setOnImageAvailableListener(this::onImageAvailable, cameraHandler);
        SurfaceTexture texture = preview.getSurfaceTexture();
        texture.setDefaultBufferSize(size.getWidth(), size.getHeight());
        previewSurface = new Surface(texture);
        try {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
            cameraManager.openCamera(activeChoice.openId, getMainExecutor(), new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice camera) {
                    cameraDevice = camera;
                    createSession();
                }
                @Override public void onDisconnected(CameraDevice camera) {
                    camera.close();
                    setStatus("Camera disconnected.");
                }
                @Override public void onError(CameraDevice camera, int error) {
                    camera.close();
                    setStatus("Could not open selected camera (Camera2 error " + error + ").");
                }
            });
            setStatus("Opening " + activeChoice.label + "…");
        } catch (CameraAccessException | SecurityException exception) {
            setStatus("Could not open selected camera: " + exception.getMessage());
            closeCamera();
        }
    }

    private Size chooseCaptureSize(CameraCharacteristics characteristics) {
        if (characteristics == null) return null;
        StreamConfigurationMap map = characteristics.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        if (map == null) return null;
        Size[] sizes = map.getOutputSizes(ImageFormat.YUV_420_888);
        if (sizes == null || sizes.length == 0) return null;
        return Arrays.stream(sizes)
                .filter(s -> s.getWidth() >= 160 && s.getHeight() >= 120)
                .min(Comparator.comparingLong(s -> Math.abs(
                        (long) s.getWidth() * s.getHeight() - 320L * 240L)))
                .orElse(sizes[0]);
    }

    private void createSession() {
        try {
            OutputConfiguration previewOutput = new OutputConfiguration(previewSurface);
            OutputConfiguration imageOutput = new OutputConfiguration(imageReader.getSurface());
            if (activeChoice.physicalId != null) {
                previewOutput.setPhysicalCameraId(activeChoice.physicalId);
                imageOutput.setPhysicalCameraId(activeChoice.physicalId);
            }
            SessionConfiguration configuration = new SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR,
                    Arrays.asList(previewOutput, imageOutput), cameraExecutor,
                    new CameraCaptureSession.StateCallback() {
                        @Override public void onConfigured(CameraCaptureSession session) {
                            if (cameraDevice == null) return;
                            captureSession = session;
                            startRepeatingRequest();
                        }
                        @Override public void onConfigureFailed(CameraCaptureSession session) {
                            setStatus("This physical lens cannot provide the requested preview stream.");
                        }
                    });
            cameraDevice.createCaptureSession(configuration);
        } catch (CameraAccessException | IllegalArgumentException exception) {
            setStatus("Could not configure selected physical lens: " + exception.getMessage());
        }
    }

    private void startRepeatingRequest() {
        try {
            requestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            requestBuilder.addTarget(previewSurface);
            requestBuilder.addTarget(imageReader.getSurface());
            requestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
            requestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON);
            requestBuilder.set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO);
            CameraCharacteristics parent = cameraManager.getCameraCharacteristics(activeChoice.openId);
            exposureCompensationRange = parent.get(
                    CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE);
            exposureCompensation = 0;
            exposureStableFrames = 0;
            exposureFrameCounter = 0;
            if (exposureCompensationRange != null && exposureCompensationRange.contains(0)) {
                requestBuilder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, 0);
            }
            Range<Integer> frameRate = chooseStableFrameRate(parent);
            if (frameRate != null) {
                requestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, frameRate);
            }
            requestBuilder.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);
            Boolean flash = parent.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
            Boolean aeLock = parent.get(CameraCharacteristics.CONTROL_AE_LOCK_AVAILABLE);
            Boolean awbLock = parent.get(CameraCharacteristics.CONTROL_AWB_LOCK_AVAILABLE);
            aeLockSupported = Boolean.TRUE.equals(aeLock);
            awbLockSupported = Boolean.TRUE.equals(awbLock);
            captureSession.setRepeatingRequest(requestBuilder.build(), null, cameraHandler);
            runOnUiThread(() -> {
                torchButton.setEnabled(Boolean.TRUE.equals(flash));
                exposureButton.setEnabled(aeLockSupported || awbLockSupported);
                connectButton.setEnabled(true);
                stopButton.setEnabled(true);
            });
            setStatus("Previewing " + activeChoice.label
                    + ". Turn on the torch and cover this lens; exposure will stabilize automatically.");
        } catch (CameraAccessException | IllegalStateException exception) {
            setStatus("Could not start preview: " + exception.getMessage());
        }
    }

    private Range<Integer> chooseStableFrameRate(CameraCharacteristics characteristics) {
        Range<Integer>[] ranges = characteristics.get(
                CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
        if (ranges == null || ranges.length == 0) return null;
        for (Range<Integer> range : ranges) {
            if (range.getLower() == 30 && range.getUpper() == 30) return range;
        }
        return Arrays.stream(ranges)
                .filter(range -> range.contains(30))
                .min(Comparator.comparingInt(range -> range.getUpper() - range.getLower()))
                .orElse(null);
    }

    private void toggleTorch() {
        if (requestBuilder == null || captureSession == null) return;
        torchEnabled = !torchEnabled;
        setExposureLock(false);
        exposureCompensation = 0;
        exposureStableFrames = 0;
        exposureFrameCounter = 0;
        if (exposureCompensationRange != null && exposureCompensationRange.contains(0)) {
            requestBuilder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, 0);
        }
        requestBuilder.set(CaptureRequest.FLASH_MODE,
                torchEnabled ? CaptureRequest.FLASH_MODE_TORCH : CaptureRequest.FLASH_MODE_OFF);
        try {
            captureSession.setRepeatingRequest(requestBuilder.build(), null, cameraHandler);
            torchButton.setText(torchEnabled ? "Torch off" : "Torch on");
            if (torchEnabled) {
                setStatus("Torch is on. Cover the selected lens; automatic exposure is searching for a clean range.");
            } else {
                setStatus("Torch is off.");
            }
        } catch (CameraAccessException exception) {
            torchEnabled = false;
            torchButton.setText("Torch on");
            setStatus("Torch is unavailable for this lens: " + exception.getMessage());
        }
    }

    private void toggleExposureLock() {
        if (requestBuilder == null || captureSession == null) return;
        setExposureLock(!exposureLocked);
        exposureStableFrames = 0;
        try {
            captureSession.setRepeatingRequest(requestBuilder.build(), null, cameraHandler);
            setStatus(exposureLocked
                    ? "Exposure and white balance are locked. Connect the USB stream and start measurement."
                    : "Automatic exposure is adjusting again.");
        } catch (CameraAccessException exception) {
            setExposureLock(false);
            setStatus("Could not change exposure lock: " + exception.getMessage());
        }
    }

    private void setExposureLock(boolean locked) {
        exposureLocked = locked;
        if (requestBuilder != null) {
            if (aeLockSupported) requestBuilder.set(CaptureRequest.CONTROL_AE_LOCK, locked);
            if (awbLockSupported) requestBuilder.set(CaptureRequest.CONTROL_AWB_LOCK, locked);
        }
        runOnUiThread(() -> exposureButton.setText(
                exposureLocked ? "Re-adjust exposure" : "Lock exposure now"));
    }

    private void updateAutoExposure(FrameStats stats) {
        if (!torchEnabled || exposureLocked || requestBuilder == null || captureSession == null) return;
        double[] ordered = stats.rgb.clone();
        Arrays.sort(ordered);
        double brightness = ordered[1];
        int channel = Math.abs(stats.rgb[0] - brightness) < 0.01 ? 0
                : Math.abs(stats.rgb[1] - brightness) < 0.01 ? 1 : 2;
        double clipping = stats.clipped[channel];
        boolean stable = brightness >= 50 && brightness <= 205 && clipping <= 0.05;
        exposureStableFrames = stable ? exposureStableFrames + 1 : 0;
        exposureFrameCounter++;

        if (exposureFrameCounter % 6 == 0 && exposureCompensationRange != null) {
            int next = exposureCompensation;
            if (clipping > 0.08 || brightness > 220) {
                next = Math.max(exposureCompensationRange.getLower(), exposureCompensation - 1);
            } else if (brightness < 40) {
                next = Math.min(exposureCompensationRange.getUpper(), exposureCompensation + 1);
            }
            if (next != exposureCompensation) {
                exposureCompensation = next;
                exposureStableFrames = 0;
                requestBuilder.set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION,
                        exposureCompensation);
                try {
                    captureSession.setRepeatingRequest(requestBuilder.build(), null, cameraHandler);
                } catch (CameraAccessException exception) {
                    setStatus("Automatic exposure adjustment failed: " + exception.getMessage());
                }
            }
        }
        if (exposureStableFrames >= 45 && (aeLockSupported || awbLockSupported)) {
            setExposureLock(true);
            try {
                captureSession.setRepeatingRequest(requestBuilder.build(), null, cameraHandler);
                setStatus("Lighting is stable and exposure is locked automatically.");
            } catch (CameraAccessException exception) {
                setExposureLock(false);
            }
        }
    }

    private void onImageAvailable(ImageReader reader) {
        Image image = reader.acquireLatestImage();
        if (image == null) return;
        try {
            FrameStats stats = analyzeImage(image);
            long timestamp = image.getTimestamp() > 0 ? image.getTimestamp() : System.nanoTime();
            updateSignalGuide(stats, timestamp);
            updateAutoExposure(stats);
            if (connected && frameInFlight.compareAndSet(false, true)) {
                networkExecutor.execute(() -> sendStats(stats, timestamp));
            }
        } finally {
            image.close();
        }
    }

    private FrameStats analyzeImage(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        Image.Plane yPlane = image.getPlanes()[0];
        Image.Plane uPlane = image.getPlanes()[1];
        Image.Plane vPlane = image.getPlanes()[2];
        ByteBuffer yBuffer = yPlane.getBuffer();
        ByteBuffer uBuffer = uPlane.getBuffer();
        ByteBuffer vBuffer = vPlane.getBuffer();
        int left = width / 5;
        int right = width * 4 / 5;
        int top = height / 5;
        int bottom = height * 4 / 5;
        double[] sum = new double[3];
        int[] clipped = new int[3];
        int count = 0;
        for (int row = top; row < bottom; row += 2) {
            for (int column = left; column < right; column += 2) {
                int y = planeValue(yBuffer, yPlane, column, row);
                int u = planeValue(uBuffer, uPlane, column / 2, row / 2);
                int v = planeValue(vBuffer, vPlane, column / 2, row / 2);
                double c = Math.max(0, y - 16);
                double d = u - 128;
                double e = v - 128;
                double red = clamp(1.164 * c + 1.596 * e);
                double green = clamp(1.164 * c - 0.392 * d - 0.813 * e);
                double blue = clamp(1.164 * c + 2.017 * d);
                sum[0] += red;
                sum[1] += green;
                sum[2] += blue;
                if (red <= 3 || red >= 252) clipped[0]++;
                if (green <= 3 || green >= 252) clipped[1]++;
                if (blue <= 3 || blue >= 252) clipped[2]++;
                count++;
            }
        }
        if (count == 0) return new FrameStats(new double[3], new double[]{1, 1, 1}, motionLevel);
        double[] means = new double[3];
        double[] clipping = new double[3];
        for (int channel = 0; channel < 3; channel++) {
            means[channel] = sum[channel] / count;
            clipping[channel] = (double) clipped[channel] / count;
        }
        return new FrameStats(means, clipping, motionLevel);
    }

    private int planeValue(ByteBuffer buffer, Image.Plane plane, int column, int row) {
        int index = buffer.position() + row * plane.getRowStride() + column * plane.getPixelStride();
        if (index < buffer.position() || index >= buffer.limit()) return 128;
        return buffer.get(index) & 0xff;
    }

    private double clamp(double value) {
        return Math.max(0, Math.min(255, value));
    }

    private void updateSignalGuide(FrameStats stats, long timestamp) {
        if (timestamp - lastGuideUpdateNs < 500_000_000L) return;
        lastGuideUpdateNs = timestamp;
        double[] ordered = stats.rgb.clone();
        Arrays.sort(ordered);
        double brightness = ordered[1];
        double bestUsefulClip = Math.min(stats.clipped[0],
                Math.min(stats.clipped[1], stats.clipped[2]));
        String light = bestUsefulClip > 0.20 ? "clipped"
                : brightness < 35 ? "too dark"
                : brightness > 220 ? "too bright" : "good";
        String message = String.format(Locale.US,
                "Live light: %s  •  RGB %.0f / %.0f / %.0f  •  best clipping %.0f%%  •  motion %.2f",
                light, stats.rgb[0], stats.rgb[1], stats.rgb[2],
                100 * bestUsefulClip, stats.motion);
        runOnUiThread(() -> signalGuide.setText(message));
    }

    private void connectNetwork() {
        int port;
        try {
            port = Integer.parseInt(portField.getText().toString());
            if (port < 1 || port > 65535) throw new NumberFormatException();
        } catch (NumberFormatException exception) {
            setStatus("Enter a valid Camera2 port, such as 9001.");
            return;
        }
        connectButton.setEnabled(false);
        setStatus("Connecting to desktop Camera2 receiver on localhost:" + port + "…");
        networkExecutor.execute(() -> {
            try {
                Socket connection = createUsbSocket(port);
                DataInputStream input = new DataInputStream(connection.getInputStream());
                DataOutputStream output = new DataOutputStream(connection.getOutputStream());
                // PPG2 is understood by both v0.3 and v0.4 desktops. Version 0.4
                // consumes its acknowledgements without blocking every camera frame.
                output.write(new byte[]{'P', 'P', 'G', '2'});
                output.flush();
                connection.setSoTimeout(0);
                socket = connection;
                socketInput = input;
                socketOutput = output;
                lastDesktopAckNs = System.nanoTime();
                connected = true;
                runOnUiThread(() -> {
                    connectButton.setEnabled(true);
                    connectButton.setText("Disconnect USB stream");
                    setStatus(exposureLocked
                            ? "Raw PPG stream connected with stable exposure. Start measurement on the desktop page."
                            : "Raw PPG stream connected. Keep the finger still while exposure adjusts.");
                });
            } catch (IOException exception) {
                closeSocket();
                runOnUiThread(() -> {
                    connectButton.setEnabled(cameraDevice != null);
                    setStatus("Could not connect. Run adb reverse tcp:" + port + " tcp:" + port
                            + " and keep the desktop server running.");
                });
            }
        });
    }

    private Socket createUsbSocket(int port) throws IOException {
        Socket connection = new Socket();
        connection.connect(new InetSocketAddress("127.0.0.1", port), 3000);
        connection.setTcpNoDelay(true);
        connection.setKeepAlive(true);
        connection.setSoTimeout(5000);
        return connection;
    }

    private void sendStats(FrameStats stats, long timestamp) {
        try {
            DataOutputStream output = socketOutput;
            DataInputStream input = socketInput;
            if (!connected || output == null || input == null) return;
            output.writeLong(timestamp);
            for (double value : stats.rgb) output.writeFloat((float) value);
            for (double value : stats.clipped) output.writeFloat((float) value);
            output.writeFloat(stats.motion);
            output.flush();
            while (input.available() > 0) {
                int acknowledgement = input.read();
                if (acknowledgement < 0) throw new IOException("desktop disconnected");
                if (acknowledgement == 1) lastDesktopAckNs = System.nanoTime();
            }
            if (System.nanoTime() - lastDesktopAckNs > DESKTOP_ACK_TIMEOUT_NS) {
                throw new SocketTimeoutException("no acknowledgement for 10 seconds");
            }
        } catch (SocketTimeoutException exception) {
            disconnectNetwork("USB stream paused: no reply from the desktop for 10 seconds. "
                    + "Check that app.py is still running, then reconnect.");
        } catch (IOException exception) {
            disconnectNetwork("USB stream lost: " + exception.getMessage());
        } finally {
            frameInFlight.set(false);
        }
    }

    private void disconnectNetwork(String message) {
        connected = false;
        closeSocket();
        runOnUiThread(() -> {
            connectButton.setText("Connect USB stream");
            connectButton.setEnabled(cameraDevice != null);
            if (message != null) setStatus(message);
        });
    }

    private synchronized void closeSocket() {
        connected = false;
        try { if (socket != null) socket.close(); } catch (IOException ignored) {}
        socket = null;
        socketInput = null;
        socketOutput = null;
        lastDesktopAckNs = 0;
        frameInFlight.set(false);
    }

    private void closeCamera() {
        disconnectNetwork(null);
        torchEnabled = false;
        exposureLocked = false;
        aeLockSupported = false;
        awbLockSupported = false;
        exposureCompensationRange = null;
        exposureCompensation = 0;
        exposureStableFrames = 0;
        exposureFrameCounter = 0;
        torchButton.setText("Torch on");
        torchButton.setEnabled(false);
        exposureButton.setText("Lock exposure now");
        exposureButton.setEnabled(false);
        signalGuide.setText("Live capture: open a lens to check light and motion.");
        connectButton.setEnabled(false);
        stopButton.setEnabled(false);
        if (captureSession != null) captureSession.close();
        if (cameraDevice != null) cameraDevice.close();
        if (imageReader != null) imageReader.close();
        if (previewSurface != null) previewSurface.release();
        captureSession = null;
        cameraDevice = null;
        imageReader = null;
        previewSurface = null;
        requestBuilder = null;
        activeChoice = null;
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (sensorManager != null && accelerometer != null) {
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME);
        }
    }

    @Override
    protected void onPause() {
        if (sensorManager != null) sensorManager.unregisterListener(this);
        haveAcceleration = false;
        motionLevel = 0;
        super.onPause();
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() != Sensor.TYPE_ACCELEROMETER || event.values.length < 3) return;
        if (haveAcceleration) {
            float dx = event.values[0] - lastAcceleration[0];
            float dy = event.values[1] - lastAcceleration[1];
            float dz = event.values[2] - lastAcceleration[2];
            float change = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
            motionLevel = 0.86f * motionLevel + 0.14f * change;
        }
        System.arraycopy(event.values, 0, lastAcceleration, 0, 3);
        haveAcceleration = true;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    private void setStatus(String message) {
        runOnUiThread(() -> status.setText(message));
    }

    @Override
    protected void onDestroy() {
        closeCamera();
        networkExecutor.shutdownNow();
        cameraThread.quitSafely();
        super.onDestroy();
    }

    private static final class CameraChoice {
        final String openId;
        final String physicalId;
        final float focalLength;
        final boolean flashAvailable;
        final float minimumFocusDistance;
        final boolean vendorDirect;
        String label;

        CameraChoice(String openId, String physicalId, float focalLength,
                     boolean flashAvailable, float minimumFocusDistance, boolean vendorDirect) {
            this.openId = openId;
            this.physicalId = physicalId;
            this.focalLength = focalLength;
            this.flashAvailable = flashAvailable;
            this.minimumFocusDistance = minimumFocusDistance;
            this.vendorDirect = vendorDirect;
        }
    }

    private static final class FrameStats {
        final double[] rgb;
        final double[] clipped;
        final float motion;

        FrameStats(double[] rgb, double[] clipped, float motion) {
            this.rgb = rgb;
            this.clipped = clipped;
            this.motion = motion;
        }
    }
}
