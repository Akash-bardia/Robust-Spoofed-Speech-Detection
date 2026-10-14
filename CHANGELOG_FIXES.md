# Changelog of Code Corrections and Scientific Refactoring

This document provides a concise, itemized before-and-after record of all architectural, mathematical, and algorithmic modifications made across the repository.

---

### 1. Acoustic Feature Pipeline: Linear Frequency Cepstral Coefficients (LFCC)
- **Before**: In `WildSpoof_Project/src/features.py`, the LFCC pipeline extracted an STFT magnitude spectrogram and simply sliced the first 60 FFT frequency bins (`spec[:60, :]`), followed by a logarithm. It did not implement a linear filterbank spanning the Nyquist spectrum and completely omitted the Discrete Cosine Transform (DCT).
- **After**: Implemented `LFCCExtractor` in `WildSpoof_Project/src/features.py` using 60 linearly spaced triangular filterbanks spanning 0–8,000 Hz, logarithmic compression of filterbank energies, and orthonormal Type-II Discrete Cosine Transform (DCT-II). The output tensor shape is rigorously asserted as `(Batch, 1, 60, 201)`.

---

### 2. Equal Error Rate (EER) and ROC-AUC Computation
- **Before**: In `WildSpoof_Project/src/evaluate.py`, EER calculation relied on SciPy's `brentq` root-finding solver or approximate nearest-neighbor ROC point searches. This resulted in numerical solver failures on flat or non-strictly monotonic ROC curves and zero-error edge cases.
- **After**: Implemented a robust closed-form monotonic linear interpolation algorithm for EER in `compute_eer`. Added unit tests in `tests/test_metrics.py` verifying precision against exact known analytic distributions.

---

### 3. Confusion Matrix and Metric Arithmetic
- **Before**: Confusion matrices contained inconsistencies between reported true/false positive counts and reported precision, recall, and accuracy numbers, caused by disparate thresholding conventions and manual reporting.
- **After**: Implemented `compute_confusion_matrix_breakdown` in `WildSpoof_Project/src/evaluate.py`. All metrics ($TN, FP, FN, TP$, Precision, Recall, Specificity, F1, Accuracy) are derived deterministically from raw prediction arrays.

---

### 4. Audio Preprocessing and Numerical Stability
- **Before**: Preprocessing routines were scattered across scripts with conflicting policies (some zero-padded, others repeated frames), and peak normalization lacked division-by-zero protection.
- **After**: Created a dedicated `WildSpoof_Project/src/preprocessing.py` standardizing utterances to exactly 32,000 samples (2.0s at 16 kHz) via central cropping (for $L > 32,000$) and symmetric zero-padding (for $L < 32,000$). Added an epsilon guard ($\epsilon = 10^{-8}$) to ensure silent or near-zero audio produces clean zeros without `NaN` or `Inf`.

---

### 5. Acoustic Augmentation and Hardening
- **Before**: Reverberation and narrowband low-pass filtering were declared as hyperparameter flags in documentation but lacked functional implementations in code. Noise SNR was not mathematically verified.
- **After**: Refactored `WildSpoof_Project/src/augmentation.py`:
  - Implemented exact signal-to-noise ratio additive Gaussian white noise: $\text{SNR}_{\text{dB}} = 10 \log_{10}(P_{\text{signal}} / P_{\text{noise}})$.
  - Implemented synthetic Room Impulse Response (RIR) reverberation: $y(t) = x(t) * h(t)$ using `scipy.signal.fftconvolve`.
  - Implemented 4th-order Butterworth low-pass filter using second-order sections (`scipy.signal.sosfilt`) with telephony cutoff (3.4 kHz).
  - Verified 2D SpecAugment frequency and time masking applied exclusively to spectrogram tensors during training.

---

### 6. Deep Neural Network Architecture & Model Checkpoints
- **Before**: Canonical documentation specified 64-D branch embeddings and 2 output classes, but serialized model weights on disk had 32-D branch embeddings and 1 output logit, resulting in `RuntimeError` during model loading.
- **After**: Updated `DualStreamLCNN` in `WildSpoof_Project/src/la_models.py` with an adaptive `load_from_checkpoint` method that inspects state dictionary dimensions and supports both 32-D (deployed checkpoint) and 64-D (canonical config) models seamlessly.

---

### 7. Model Size and Memory Accounting
- **Before**: Documentation conflated the 349 KB serialized `.pt` checkpoint file on disk with the active model parameter count.
- **After**: Added `get_model_summary` in `WildSpoof_Project/src/la_models.py` to independently profile trainable parameters (84,705), FP32 parameter memory (331.38 KB), FP16 parameter memory (165.69 KB), and disk file size (349,190 bytes).

---

### 8. Terminology and Scientific Classification
- **Before**: In-distribution noise levels (5, 10, 20 dB) were mislabeled as "unseen acoustic conditions," Gaussian noise was called "ambient environmental noise," and cross-dataset evaluation on `for-2seconds` was claimed as a completed objective despite severe domain collapse.
- **After**: Corrected terminology across all documentation:
  - Used "controlled acoustic stress conditions" for corruption evaluations.
  - Used "additive Gaussian white noise" for noise augmentation.
  - Used "unseen attack algorithms" for evaluation attacks $A07$–$A19$.
  - Reclassified cross-dataset evaluation as "Preliminary Exploration / Future Work."

---

### 9. Unified Automated Results Generator
- **Before**: Evaluation scripts output disconnected CSV files and figures, leading to documentation drift.
- **After**: Created `WildSpoof_Project/scripts/generate_results.py` to run end-to-end evaluation from model checkpoints and raw audio, generating all summary CSVs in `WildSpoof_Project/results/metrics/` and high-resolution figures in `WildSpoof_Project/results/figures/`.

---

### 10. Automated Unit Test Suite
- **Before**: No automated unit test suite existed.
- **After**: Built `tests/` comprising 12 automated unit tests using `pytest` covering preprocessing, feature extractors, MFM layers, augmentation, metrics, and a 10-sample end-to-end overfitting sanity test (`test_tiny_overfit.py`). All 12 unit tests pass in 14.12s.
