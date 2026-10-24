# Architecture and working mechanism

```text
UCI AI4I CSV
   |
   v
Training pipeline (pandas + scikit-learn)
   |-- schema checks / feature selection
   |-- stratified train-test split
   |-- preprocessing (imputation + one-hot encoding)
   |-- Random Forest classifier
   |-- Isolation Forest anomaly detector
   |-- evaluation metrics + serialized artifacts
   v
FastAPI inference service
   |-- Pydantic input validation
   |-- model probability
   |-- configurable risk threshold
   |-- anomaly score
   v
React dashboard
   |-- sensor controls
   |-- risk result
   |-- anomaly result
   |-- evaluation metrics
```

## Classification vs anomaly detection
- Failure classifier: estimates `P(machine failure = 1 | sensor snapshot)` based on labeled training examples.
- Anomaly detector: flags sensor combinations that differ from the training distribution. It does not say that a failure will definitely happen.
- Threshold: turns a probability into a UI label. Tune it on a validation set based on the relative cost of false negatives and false positives; do not choose it based on the test set.

## Current limitations
- No actual sensor connection or streaming transport.
- No persistent database or alert delivery.
- No RUL estimate.
- Dataset is synthetic and one row represents an operating sample, not a long per-machine trajectory.
