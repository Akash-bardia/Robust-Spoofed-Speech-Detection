# WildSpoof: Robust Spoofed Speech Detection (ICASSP GC-14 Inspired)

A lightweight deep learning anti-spoofing countermeasure architecture designed for the canonical **ASVspoof 2019 Logical Access (`LA`)** challenge and hardened for controlled acoustic mismatch.

---

## 1. Executive Summary & Core Objectives

Synthetic speech generators (TTS) and voice conversion (VC) models pose a grave threat to voice biometrics. While clean laboratory detectors achieve low Equal Error Rates (EER), acoustic degradation (additive noise, reverberation, and channel bandwidth limitations) degrades baseline countermeasure reliability.

This repository provides:
- **Canonical ASVspoof 2019 Logical Access (`LA`)**: 121,461 audio files across 19 attack systems (`A01`–`A19`), with strictly disjoint speaker and attack splits.
- **Dual-Stream MFM-LCNN**: Parallel convolutional encoders processing 60-band Linear Frequency Cepstral Coefficients (LFCC) and 80-band Log-Mel Spectrograms with Max-Feature-Map (MFM) competitive non-linearities.
- **Mathematically Verified Acoustic Features**: Genuine linear filterbanks (0–8,000 Hz) + DCT-II for LFCC, avoiding any FFT-bin truncation.
- **Acoustic Hardening**: Dual-domain data augmentation combining time-domain additive Gaussian noise, synthetic room impulse response (RIR) reverberation, 4th-order Butterworth low-pass filtering, and 2D SpecAugment.
- **Controlled Acoustic Stress Testing**: Systematic comparison of Clean-trained Baseline vs. Acoustically Augmented models under 7 controlled acoustic conditions.

---

## 2. Model Size & Complexity Clarification

| Configuration | Branch Embedding | Fused Dimension | Output Logits | Trainable Parameters | FP32 Memory | Serialized Disk Checkpoint Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Deployed Checkpoint (Baseline)** | 32 | 64 | 1 | **84,705** | **331.38 KB** | **349,190 bytes (~341 KB)** |
| **Deployed Checkpoint (Augmented)**| 32 | 64 | 1 | **84,705** | **331.38 KB** | **349,221 bytes (~341 KB)** |
| **Canonical Config Architecture**  | 64 | 128 | 2 | **125,858**| **492.14 KB** | **~512 KB** |

*Note*: The "349 KB" value documented in project milestones represents the **serialized file size on disk** (`.pt`), whereas the model's actual parameter footprint is **84,705 parameters** (331.38 KB in FP32).

---

## 3. Key Empirical Results (ASVspoof 2019 LA Evaluation Set)

All metrics are generated directly from model inference via `scripts/generate_results.py`:

### 3.1. Overall Evaluation (2,995 Utterances)
- **Baseline LCNN**: ROC-AUC **0.9129**, EER **16.13%** (operating threshold 0.9836).
- **Augmented LCNN**: ROC-AUC **0.8962**, EER **17.86%** (operating threshold 0.6685).

### 3.2. Controlled Acoustic Stress Testing (500 Utterances Paired Evaluation)

| Acoustic Condition | Baseline AUC | Baseline EER (%) | Augmented AUC | Augmented EER (%) | AUC Lift ($\Delta$) | EER Reduction (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pristine Clean** | **0.9140** | **14.80%** | 0.8980 | 17.60% | -0.0160 | -2.80% |
| **Gaussian Noise 20 dB** | 0.7019 | 36.00% | **0.7768** | **28.80%** | **+0.0749** | **+7.20%** |
| **Gaussian Noise 10 dB** | 0.4885 | 52.00% | **0.6965** | **35.60%** | **+0.2080** | **+16.40%** |
| **Gaussian Noise 5 dB** | 0.4684 | 54.00% | **0.6712** | **36.80%** | **+0.2028** | **+17.20%** |
| **Reverberation (Synthetic RIR)** | 0.7428 | 33.20% | **0.8758** | **20.80%** | **+0.1330** | **+12.40%** |
| **Noise + Reverberation** | 0.3574 | 60.40% | **0.6312** | **40.80%** | **+0.2738** | **+19.60%** |
| **Lowpass Filter (3.4 kHz)** | 0.8487 | 23.20% | **0.8492** | **23.20%** | **+0.0005** | 0.00% |

Under severe acoustic corruption (5 dB noise and combined noise + reverb), the baseline collapses ($\text{AUC} \le 0.4684$), while the acoustically hardened model preserves robust discriminability, delivering up to **+0.2738 AUC lift** and reducing EER by **19.60%**.

---

## 4. Execution Pipelines

```powershell
# 1. Run unit test suite
pytest -v tests/

# 2. Regenerate all empirical result CSVs and publication figures
python WildSpoof_Project\scripts\generate_results.py

# 3. Controlled acoustic stress-testing
python WildSpoof_Project\scripts\evaluate_la_robustness.py

# 4. Granular unseen attack evaluation (A07-A19)
python WildSpoof_Project\scripts\evaluate_la_attacks.py
```
