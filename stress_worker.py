"""Private JSON-lines CPU inference worker, using the training kernel's environment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from stress_input import PhoneCaptureRejected, prepare_phone_capture


def send(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    try:
        bundle = Path(args.bundle).resolve()
        checks = json.loads((bundle / "checksums.json").read_text())
        for name in ("stress_runtime.py", "stress_model.pt", "model_config.json"):
            if name not in checks or hashlib.sha256((bundle / name).read_bytes()).hexdigest() != checks[name]:
                raise ValueError(f"Bundle checksum failed: {name}; restore the exported bundle")
        import torch
        torch.set_num_threads(2)
        spec = importlib.util.spec_from_file_location("pulse_trained_stress", bundle / "stress_runtime.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        predictor = runtime.StressPredictor(bundle, device="cpu")
        if predictor.config.get("input_seconds_with_context") != 64:
            raise ValueError("This monitor requires the trained 64-second input contract")
        send({"type": "ready", "threshold": predictor.config["threshold"],
              "pipeline_version": predictor.config["pipeline_version"]})
    except Exception as exc:
        send({"type": "error", "reason": f"Model could not load: {type(exc).__name__}: {exc}"})
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("type") != "predict":
                raise ValueError("Unknown worker request")
            raw, fs, capture = prepare_phone_capture(request["rows"], request["channel"])
            result = predictor.predict_raw(raw, fs)
            if capture["extended_frame_repair"]:
                # The classifier uses waveform samples. Pulse variability
                # estimates depend on individual beat timing.
                result["signal_diagnostics"].update(
                    prv_valid=False,
                    prv_reason="Frame repair exceeds the pulse-variability timing budget",
                    mean_hr_bpm=None, rmssd_ms=None, sdnn_ms=None,
                )
            result["capture_diagnostics"] = capture
            send({"type": "result", "result": result})
        except (runtime.SignalRejected, PhoneCaptureRejected) as exc:
            send({"type": "rejected", "reason": str(exc)})
        except Exception as exc:
            send({"type": "error", "reason": f"Inference failed: {type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
