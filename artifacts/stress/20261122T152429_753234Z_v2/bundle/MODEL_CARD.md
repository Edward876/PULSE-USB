# Experimental PPG stress classifier — v2

Architecture: one six-block multi-scale residual CNN with a learned frequency branch.
Input: 64 seconds raw PPG; central 60 s at 30 Hz; normalized signal + derivative.
No PRV/HRV, ECG, EDA, temperature, subject ID, timestamps or quality flags are classifier inputs.
PRV is reported separately only if bounded interval and morphology checks pass.
MSPTDfast cannot recover physiological pulse information absent from a noisy signal.

Fitted subjects: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]. Four development folds selected checkpoints and threshold.
Final fit: 14 epochs. Decision threshold: 0.31.
Test subjects excluded from fitting: [15, 16, 17].
Evaluation status: exploratory_reuse_of_previous_test_subjects.
These same test people were inspected in v1; these results are exploratory, not independent confirmation.
Overlapping windows are correlated; window count is not participant count.

Accuracy: 0.8352
Macro-F1: 0.8318
Subject-mean macro-F1: 0.8318
Stress precision: 0.6981
Stress recall: 0.9578
ROC-AUC: 0.9064

The sigmoid output is an uncalibrated baseline/stress class score, not a personal stress
probability, severity measurement or disease diagnosis. WESAD is a wrist-sensor lab
dataset. Its stress protocol includes movement; waveform classifiers can learn motion
or protocol artifacts. Corrected PRV gating removes corrupted PRV inputs but does not
prove removal of all waveform confounds. Phone fingertip transfer is not established.
Independent labelled phone recordings and new participants are required for validation.

The live app retains only 24 seconds. Integration needs a real 64-second raw buffer,
stable channel/exposure, frame timestamps and capture-quality checks. Do not repeat or
pad a 24-second report to pretend it contains 64 seconds. Live app code is unchanged.

Load: import StressPredictor from stress_runtime, then predict_raw(raw, actual_fs),
or predict_phone_rows(rows, channel='green'). Keep the entire bundle together.
SignalRejected means insufficient/invalid acquisition. PRV can be unavailable even when
a waveform classifier score is returned; that score must remain marked experimental.

Research adaptations, not exact reproduced benchmarks or pretrained weights:
Charlton et al. (2025), doi:10.1088/1361-6579/adb89e (MSPTDfast);
Rashid et al. (2021), doi:10.1109/EMBC46164.2021.9630576 (PPG stress hybrid CNN);
Fawaz et al. (2020), doi:10.1007/s10618-020-00710-y (InceptionTime).
Dataset: WESAD, doi:10.24432/C57K5T. Follow the publisher's readme and cite its paper.
