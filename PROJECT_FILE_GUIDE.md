# Project File Guide & Architecture Blueprint

**Project Title**: Robust Spoofed Speech Detection: Bona Fide vs. Spoof Speech under Acoustic Mismatch  
**Dataset**: ASVspoof 2019 Logical Access (`LA`)  
**Architecture**: Dual-Stream MFM-LCNN (LFCC + Log-Mel)  
**Objective**: Robust speech anti-spoofing under clean and controlled acoustic stress conditions.

---

## Reviewer Quick Start

If you have only **5 minutes** to inspect the repository, review the core files in this sequence:

1. **Project Specification & Benchmark**: [`README.md`](file:///d:/Speech/PROJECT/README.md)  
   *Overview of project motivation, architecture, results, and step-by-step reproduction commands.*
2. **Final Academic Presentation**: [`Speech_AntiSpoofing_Final_Presentation_Final.pptx`](file:///d:/Speech/PROJECT/Speech_AntiSpoofing_Final_Presentation_Final.pptx)  
   *21-slide visual presentation containing verified experimental findings and architecture diagrams.*
3. **Core Neural Architecture**: [`WildSpoof_Project/src/la_models.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/la_models.py)  
   *Max-Feature-Map (MFM) activation layer, 4-stage LCNN convolutional encoders, and DualStreamLCNN.*
4. **Mathematical LFCC & Mel Extraction**: [`WildSpoof_Project/src/features.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/features.py)  
   *60 linear triangular filterbanks spanning 0–8,000 Hz, orthonormal DCT-II, and 80-band Log-Mel extractor.*
5. **Signal Preprocessing & Standardization**: [`WildSpoof_Project/src/preprocessing.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/preprocessing.py)  
   *16 kHz standardization, fixed 2.0-second windowing (exactly 32,000 samples), and epsilon-safe normalization.*
6. **Acoustic Augmentation Engine**: [`WildSpoof_Project/src/augmentation.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/augmentation.py)  
   *Additive Gaussian noise (SNR power calculation), synthetic RIR reverberation, Butterworth lowpass filtering, and 2D SpecAugment.*
7. **Model Training Script**: [`WildSpoof_Project/scripts/train_la_system.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/scripts/train_la_system.py)  
   *Complete training pipeline with weighted BCE loss, Adam optimizer, cosine annealing, and validation EER checkpointing.*
8. **Evaluation & Monotonic EER Metrics**: [`WildSpoof_Project/src/evaluate.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/evaluate.py)  
   *Closed-form monotonic linear interpolation for Equal Error Rate (EER), ROC-AUC, and confusion matrix breakdown.*
9. **Results & Synchronized Figures**: [`WildSpoof_Project/results/`](file:///d:/Speech/PROJECT/WildSpoof_Project/results/)  
   *Raw evaluation CSV tables in `results/metrics/` and high-resolution figures in `results/figures/`.*
10. **Run Automated Test Suite**: Run `pytest -v tests/`  
    *12 automated unit tests verifying preprocessing, features, MFM, augmentation, metrics, and a 10-sample overfitting test.*

---

## File & Directory Guide

| File / Directory Path | Purpose | Category | Referenced / Used By |
| :--- | :--- | :--- | :--- |
| [`README.md`](file:///d:/Speech/PROJECT/README.md) | Master entry-point documentation with full methodology and metrics. | Documentation | Reviewers, Users |
| [`PROJECT_FILE_GUIDE.md`](file:///d:/Speech/PROJECT/PROJECT_FILE_GUIDE.md) | Complete reviewer guide, file breakdown, and pipeline map. | Documentation | Reviewers |
| [`AUDIT_REPORT.md`](file:///d:/Speech/PROJECT/AUDIT_REPORT.md) | Exhaustive 30-phase scientific and code audit report. | Audit Documentation | Reviewers, Auditors |
| [`CHANGELOG_FIXES.md`](file:///d:/Speech/PROJECT/CHANGELOG_FIXES.md) | Itemized before/after record of code fixes and mathematical corrections. | Documentation | Reviewers |
| [`requirements.txt`](file:///d:/Speech/PROJECT/requirements.txt) | Unified Python package dependencies. | Configuration | `pip install -r requirements.txt` |
| [`Speech_AntiSpoofing_Final_Presentation_Final.pptx`](file:///d:/Speech/PROJECT/Speech_AntiSpoofing_Final_Presentation_Final.pptx) | Final 21-slide academic presentation deck. | Presentation | Defense / Review |
| [`tests/`](file:///d:/Speech/PROJECT/tests/) | Automated unit test suite (12 tests covering all components). | Verification | `pytest -v tests/` |
| [`LA/`](file:///d:/Speech/PROJECT/LA/) | Canonical ASVspoof 2019 Logical Access dataset (audio & protocols). | Benchmark Dataset | Manifest scripts, Datasets |
| [`WildSpoof_Project/src/config.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/config.py) | Central configuration parameters, paths, and hyperparameters. | Configuration | All scripts and modules |
| [`WildSpoof_Project/src/preprocessing.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/preprocessing.py) | Audio loading, 16 kHz resampling, 32,000-sample windowing, safe normalization. | Preprocessing | `la_dataset.py`, `tests/` |
| [`WildSpoof_Project/src/features.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/features.py) | True 60-filterbank LFCC with DCT-II and 80-band Log-Mel extractor. | Feature Extraction | `la_dataset.py`, `train_la_system.py`, `tests/` |
| [`WildSpoof_Project/src/augmentation.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/augmentation.py) | Additive Gaussian noise, synthetic RIR reverberation, lowpass, SpecAugment. | Augmentation | `la_dataset.py`, `evaluate_la_robustness.py` |
| [`WildSpoof_Project/src/la_models.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/la_models.py) | MFM competitive activation, LCNNEncoder, DualStreamLCNN architecture. | Model Definition | All training/evaluation scripts |
| [`WildSpoof_Project/src/models.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/models.py) | Clean canonical alias importing from `la_models.py`. | Model Alias | External / Test entry |
| [`WildSpoof_Project/src/la_dataset.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/la_dataset.py) | PyTorch Dataset for ASVspoof 2019 LA protocol manifests and audio. | Dataset Loader | `train_la_system.py`, `evaluate_*.py` |
| [`WildSpoof_Project/src/dataset.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/dataset.py) | Clean canonical alias importing from `la_dataset.py`. | Dataset Alias | External / Test entry |
| [`WildSpoof_Project/src/evaluate.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/evaluate.py) | Linear interpolation EER, ROC-AUC, confusion matrix breakdown. | Evaluation / Metrics | All evaluation scripts |
| [`WildSpoof_Project/src/utils.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/src/utils.py) | Random seed and deterministic environment utilities. | Utility | Training / Test scripts |
| [`WildSpoof_Project/scripts/prepare_la_manifests.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/scripts/prepare_la_manifests.py) | Parses official ASVspoof protocols and outputs CSV manifests. | Data Preparation | Pipeline step 1 |
| [`WildSpoof_Project/scripts/train_la_system.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/scripts/train_la_system.py) | End-to-end training script for Baseline and Augmented LCNN models. | Model Training | Pipeline step 2 |
| [`WildSpoof_Project/scripts/evaluate_la_attacks.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/scripts/evaluate_la_attacks.py) | Granular evaluation across 13 unseen evaluation attacks (A07–A19). | Evaluation | Pipeline step 3 |
| [`WildSpoof_Project/scripts/evaluate_la_robustness.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/scripts/evaluate_la_robustness.py) | Controlled acoustic stress testing across 7 conditions. | Robustness Study | Pipeline step 4 |
| [`WildSpoof_Project/scripts/generate_results.py`](file:///d:/Speech/PROJECT/WildSpoof_Project/scripts/generate_results.py) | Unified 100% data-driven regenerator for all CSV metrics and figures. | Result Generation | Pipeline step 5 |
| [`WildSpoof_Project/models/`](file:///d:/Speech/PROJECT/WildSpoof_Project/models/) | Retained final trained model checkpoints (~349 KB each). | Checkpoints | Evaluation scripts |
| [`WildSpoof_Project/results/metrics/`](file:///d:/Speech/PROJECT/WildSpoof_Project/results/metrics/) | Generated CSV evaluation tables. | Results | Reports / Defense |
| [`WildSpoof_Project/results/figures/`](file:///d:/Speech/PROJECT/WildSpoof_Project/results/figures/) | Publication-ready evaluation plots and confusion matrices. | Figures | Reports / Presentation |

---

## Final Execution Flow & Pipeline Map

```text
========================================================================================
                                     DATA INGESTION
========================================================================================
ASVspoof 2019 Logical Access (LA) Dataset
├── ASVspoof2019_LA_cm_protocols/ (ASVspoof2019.LA.cm.train/dev/eval.trl.txt)
└── Audio FLAC files (train: 25,380 | dev: 24,844 | eval: 71,237)
                                │
                                ▼
scripts/prepare_la_manifests.py ──► Generates manifests:
                                    ├── data/manifests/train_la.csv
                                    ├── data/manifests/dev_la.csv
                                    └── data/manifests/eval_la.csv

========================================================================================
                              TRAINING & FEATURE PIPELINE
========================================================================================
src/la_dataset.py (ASVspoofLADataset)
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
src/preprocessing.py                            src/augmentation.py (Augmented Only)
- 16 kHz Mono Resampling                        - Additive Gaussian Noise (SNR: 5–25 dB)
- Fixed 2.0s Window (32,000 samples)            - Synthetic RIR Reverberation
- Peak Normalization ([-1, 1], eps=1e-8)        - 4th-Order Butterworth Lowpass (3.4 kHz)
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
src/features.py
├── LFCCExtractor: STFT Power -> 60 Linear Filterbanks (0-8 kHz) -> Log -> DCT-II -> (1, 60, 201)
└── LogMelExtractor: STFT Power -> 80 Mel Bands (0-8 kHz) -> AmplitudeToDB -> (1, 80, 201)
                                │
                                ▼
src/la_models.py (DualStreamLCNN)
├── LFCC Branch: 4-Stage MFM Conv Encoder (Conv+MFM+Pool) -> 32-D/64-D Latent
├── Log-Mel Branch: 4-Stage MFM Conv Encoder (Conv+MFM+Pool) -> 32-D/64-D Latent
├── Latent Fusion: Concatenate (64-D/128-D) + BatchNorm + Dropout(0.35)
└── Classifier Head: Linear -> Logits
                                │
                                ▼
scripts/train_la_system.py
├── Loss: Class-Weighted Binary Cross-Entropy
├── Optimizer: Adam (initial LR = 3e-4) + CosineAnnealingLR
├── Gradient Norm Clipping: max_norm = 1.0
└── Checkpoints Saved:
    ├── models/la_academic_lcnn_baseline.pt  (Trained on clean audio)
    └── models/la_academic_lcnn_augmented.pt (Trained with acoustic hardening)

========================================================================================
                               EVALUATION & INFERENCE
========================================================================================
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
scripts/generate_results.py  scripts/evaluate_la_attacks.py  scripts/evaluate_la_robustness.py
(Unified Evaluation)         (13 Unseen Attacks A07–A19)      (7 Stress Conditions)
        │                       │                       │
        ▼                       ▼                       ▼
src/evaluate.py
├── Monotonic Linear Interpolation EER (compute_eer)
├── ROC-AUC Score Calculation (compute_roc_auc)
└── Confusion Matrix Breakdown (compute_confusion_matrix_breakdown)
        │
        ▼
========================================================================================
                                   VERIFIED OUTPUTS
========================================================================================
results/metrics/
├── la_eval_overall.csv          (Overall metrics: AUC 0.9129, EER 16.13%)
├── la_attack_breakdown.csv      (A07–A19 granular attack rankings)
├── la_robustness_comparison.csv (Paired clean vs. stress results)
└── model_size_summary.csv       (84,705 params, 331 KB FP32 vs. 349 KB disk file)

results/figures/
├── 04_confusion_matrix_evaluation.png        (LA evaluation subset confusion matrix)
├── 05_roc_curves_official_la.png             (ROC curves for baseline vs. augmented)
├── 06_unseen_attacks_eer_barchart.png        (Attack-wise EER comparison)
├── 07_unseen_attacks_auc_barchart.png        (Attack-wise AUC comparison)
└── 08_acoustic_robustness_stress_testing.png (Robustness lifts across 7 conditions)
```

---

## Detailed Component Documentation

### 1. `WildSpoof_Project/src/preprocessing.py`
- **Role**: Audio input standardization.
- **Why Necessary**: Guarantees that every raw audio file conforms to the uniform tensor specifications expected by downstream feature extractors and GPU batch loaders.
- **Key Functions**:
  - `standardize_audio(audio, sr, target_sr=16000, target_duration=2.0)`: Converts multi-channel audio to mono, resamples to 16 kHz, centrally crops files $>2.0$s, and symmetrically zero-pads files $<2.0$s to exactly 32,000 samples.
  - `peak_normalize(audio, eps=1e-8)`: Normalizes amplitude to $[-1.0, 1.0]$ with an epsilon guard, preventing clipping and eliminating `NaN` or `Inf` values from silent recordings.
- **Used By**: `src/la_dataset.py`, `tests/test_preprocessing.py`.

### 2. `WildSpoof_Project/src/features.py`
- **Role**: Mathematical acoustic feature extraction.
- **Why Necessary**: Deep speech anti-spoofing countermeasures depend on frequency representations that preserve high-frequency synthesis cues (LFCC) alongside perceptual vocal tract formant dynamics (Log-Mel).
- **Key Classes**:
  - `LFCCExtractor`: Computes STFT power spectrum, applies 60 linearly spaced triangular filterbanks covering the full 0–8,000 Hz spectrum, calculates logarithmic energies, and performs orthonormal Type-II Discrete Cosine Transform (DCT-II). Output shape: `(Batch, 1, 60, 201)`.
  - `LogMelExtractor`: Computes STFT power spectrum, applies 80 Mel-scale filterbanks (0–8,000 Hz), and converts to decibels via `AmplitudeToDB`. Output shape: `(Batch, 1, 80, 201)`.
- **Used By**: `src/la_dataset.py`, `tests/test_features.py`, `tests/test_tiny_overfit.py`.

### 3. `WildSpoof_Project/src/augmentation.py`
- **Role**: Time-domain and spectrogram acoustic hardening.
- **Why Necessary**: Countermeasures trained solely on pristine laboratory recordings collapse under environmental mismatch; this module hardens the network against noise, room acoustics, and channel constraints.
- **Key Functions**:
  - `add_gaussian_noise(audio, snr_db)`: Mathematically injects Gaussian white noise according to $P_{\text{noise}} = P_{\text{signal}} / 10^{\text{SNR} / 10}$.
  - `apply_reverberation(audio, decay_rate, rir_duration=0.2)`: Convolves speech with an exponential room impulse response $y(t) = x(t) * h(t)$ via FFT convolution.
  - `apply_lowpass_filter(audio, cutoff_freq)`: 4th-order Butterworth low-pass filter via second-order sections (`sosfilt`) modeling telephony bandwidth limits (3.4 kHz).
  - `apply_specaugment(spec, max_f, max_t)`: 2D rectangular zero-masking on frequency channels and time frames during training.
- **Used By**: `src/la_dataset.py`, `scripts/evaluate_la_robustness.py`, `tests/test_augmentations.py`.

### 4. `WildSpoof_Project/src/la_models.py`
- **Role**: Deep neural network model definitions.
- **Why Necessary**: Implements the core Dual-Stream Light CNN (LCNN) architecture and Max-Feature-Map (MFM) non-linear activation.
- **Key Classes**:
  - `MFM(nn.Module)`: Computes $\max(X_1, X_2)$ across channel halves, performing dynamic competitive feature selection that suppresses low-energy noise channels.
  - `LCNNEncoder(nn.Module)`: 4-stage convolutional encoder with MFM non-linearities, max pooling, and adaptive average pooling.
  - `DualStreamLCNN(nn.Module)`: Parallel dual-branch architecture extracting 32-D/64-D embeddings from LFCC and Log-Mel, fusing them into a 64-D/128-D latent representation before dense classification.
  - `get_model_summary(model, file_path)`: Computes exact trainable parameters, FP32/FP16 memory footprint, and serialized checkpoint file size.
- **Used By**: All training, evaluation, and test scripts.

### 5. `WildSpoof_Project/src/la_dataset.py`
- **Role**: Protocol parsing and PyTorch Dataset loading.
- **Why Necessary**: Connects disk audio files to the training and evaluation loops, executing on-the-fly feature extraction and augmentation without memory bottlenecks.
- **Key Classes**:
  - `ASVspoofLADataset(Dataset)`: Reads manifest CSVs, loads audio via `sf.read`, executes preprocessing, applies training augmentations, extracts dual LFCC + Mel tensors, and returns labeled batches.
- **Used By**: `scripts/train_la_system.py`, `scripts/evaluate_la_attacks.py`, `scripts/evaluate_la_robustness.py`, `scripts/generate_results.py`.

### 6. `WildSpoof_Project/src/evaluate.py`
- **Role**: Academic metric computation and inference.
- **Why Necessary**: Anti-spoofing performance must be evaluated using threshold-independent criteria (Equal Error Rate and ROC-AUC) due to extreme class imbalance (~9:1 spoof-to-bona fide ratio).
- **Key Functions**:
  - `compute_eer(labels, scores)`: Robust closed-form monotonic linear interpolation along the ROC curve to find the exact point where False Acceptance Rate equals False Rejection Rate.
  - `compute_roc_auc(labels, scores)`: Computes trapezoidal area under the receiver operating characteristic curve.
  - `compute_confusion_matrix_breakdown(labels, preds)`: Derives exact counts ($TN, FP, FN, TP$), row-normalized percentages, precision, recall, specificity, F1, and accuracy.
- **Used By**: All evaluation scripts, `tests/test_metrics.py`.

### 7. `WildSpoof_Project/scripts/prepare_la_manifests.py`
- **Role**: Protocol ingestion and manifest generation.
- **Why Necessary**: Parses the canonical ASVspoof 2019 LA protocol text files (`ASVspoof2019.LA.cm.train/dev/eval.trl.txt`) and produces structured, reproducible CSV manifests (`train_la.csv`, `dev_la.csv`, `eval_la.csv`) with validated audio paths, speaker IDs, attack algorithms, and binary labels.
- **Used By**: Manual execution / First-time setup.

### 8. `WildSpoof_Project/scripts/train_la_system.py`
- **Role**: Complete model training and checkpointing.
- **Why Necessary**: Trains both the Clean Baseline LCNN and the Acoustically Hardened Augmented LCNN models, applying class-weighted cross-entropy loss, Adam optimization with cosine annealing decay, gradient clipping, and validation EER checkpointing.
- **Used By**: Reproducing model training.

### 9. `WildSpoof_Project/scripts/evaluate_la_attacks.py`
- **Role**: Granular unseen attack evaluation ($A07$–$A19$).
- **Why Necessary**: Evaluates trained models across all 13 unseen evaluation attack algorithms, producing granular EER and ROC-AUC metrics and identifying the easiest and most challenging speech synthesis systems.
- **Used By**: Academic benchmarking.

### 10. `WildSpoof_Project/scripts/evaluate_la_robustness.py`
- **Role**: Controlled acoustic stress testing.
- **Why Necessary**: Executes paired robustness evaluations across 7 controlled conditions (Clean, Gaussian Noise 20/10/5 dB, Reverberation, Combined Noise+Reverb, Lowpass 3.4 kHz) on identical speech utterances to quantify the exact robustness lift of acoustic hardening.
- **Used By**: Robustness study verification.

### 11. `WildSpoof_Project/scripts/generate_results.py`
- **Role**: Unified data-driven results regenerator.
- **Why Necessary**: Runs end-to-end evaluation from model checkpoints and raw audio, generating all summary CSVs in `results/metrics/` and high-resolution figures in `results/figures/` without manual transcription.
- **Used By**: Results reproduction and verification.

---

## Retained Model Checkpoints

Both retained checkpoints are stored in [`WildSpoof_Project/models/`](file:///d:/Speech/PROJECT/WildSpoof_Project/models/):

| Checkpoint File | Architecture | Training Condition | Trainable Params | FP32 Memory | Serialized Disk Size | Role in Study |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `la_academic_lcnn_baseline.pt` | DualStreamLCNN (32-D branch, 64-D fused, 1 logit) | Clean ASVspoof 2019 LA Train | **84,705** | **331.38 KB** | **349,190 bytes (~341 KB)** | Baseline countermeasure model representing clean laboratory training. |
| `la_academic_lcnn_augmented.pt`| DualStreamLCNN (32-D branch, 64-D fused, 1 logit) | Hardened with Noise, Reverb, Lowpass, SpecAugment | **84,705** | **331.38 KB** | **349,221 bytes (~341 KB)** | Acoustically hardened countermeasure model demonstrating robustness lifts under stress. |

---

## Automated Verification Suite

The repository includes 12 automated unit tests in [`tests/`](file:///d:/Speech/PROJECT/tests/), executable via:
```powershell
pytest -v tests/
```

- `test_augmentations.py::test_gaussian_noise_snr`: Confirms empirical noise power matches theoretical SNR formula within 0.75 dB.
- `test_augmentations.py::test_reverberation_convolution`: Verifies temporal dispersion and energy decay of synthetic RIR convolution.
- `test_augmentations.py::test_lowpass_filter_attenuation`: Asserts $>25$ dB high-frequency attenuation for Butterworth low-pass filter.
- `test_augmentations.py::test_specaugment_masking`: Verifies 2D time and frequency zero-masking during training.
- `test_features.py::test_lfcc_mathematical_properties`: Asserts 60 linear triangular filters spanning 0–8,000 Hz, DCT-II decorrelation, and tensor shape `(1, 1, 60, 201)`.
- `test_features.py::test_logmel_properties`: Asserts 80 Mel bands, amplitude-to-dB conversion, and tensor shape `(1, 1, 80, 201)`.
- `test_metrics.py::test_eer_monotonic_interpolation`: Verifies exact closed-form interpolation on known analytic distributions.
- `test_metrics.py::test_confusion_matrix_breakdown`: Asserts integer count conservation and rate arithmetic.
- `test_mfm_and_models.py::test_mfm_activation`: Verifies $\max(X_1, X_2)$ channel halving and odd-channel assertions.
- `test_mfm_and_models.py::test_lcnn_encoder_and_dual_stream`: Asserts end-to-end forward pass and backward autograd gradient flow.
- `test_preprocessing.py::test_preprocessing_durations_and_channels`: Asserts 1s, 2s, 3s, stereo, silence, and near-zero waveforms yield exact $(1, 32000)$ tensors without `NaN` or `Inf`.
- `test_tiny_overfit.py::test_tiny_overfit_pipeline`: Verifies the model rapidly overfits a 10-sample synthetic batch, reducing loss by $>90\%$.
