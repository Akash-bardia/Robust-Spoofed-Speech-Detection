# WildSpoof: In-the-Wild Generation and Detection of Spoofed Speech (GC-14)

**Challenge Details**:
- **Grand Challenge**: ICASSP GC-14: In-the-wild generation and detection of spoofed speech (WildSpoof)
- **Organizers**: Yihan Wu (Renmin University), Jee-weon Jung (Apple / CMU), Hye-jin Shim (CMU), Xin Wang (NII Japan), Xin Cheng (Renmin University)
- **Challenge Website**: [https://wildspoof.github.io/](https://wildspoof.github.io/)
- **Submission Portal**: [ICASSP Workshop Paper Submission](https://cmsworkshops.com/ICASSP2026/Papers/Submission.asp?Type=Challenge&ID=14)
- **Target Tasks**: Text-to-Speech Generation (TTS) & Spoofing-aware Automatic Speaker Verification (SASV)

---

## 1. Executive Summary & Main Objectives

The WildSpoof Challenge aims to advance the use of in-the-wild data in two speech processing tasks that generate and detect spoofed speech:
1. **Text-to-Speech Generation (TTS)**
2. **Spoofing-aware Automatic Speaker Verification (SASV)**

This project addresses the detection side of the WildSpoof Challenge, moving beyond conventional clean and controlled laboratory datasets toward realistic, in-the-wild robustness. Evaluated primarily on the canonical **ASVspoof 2019 Logical Access (`LA`)** benchmark dataset (121,461 utterances across 19 speech synthesis & voice conversion attack algorithms `A01`–`A19`), our architecture incorporates acoustic noise and reverberation hardening to detect spoofed speech under both pristine and degraded real-world conditions.

```text
BONA FIDE (HUMAN SPEECH)   vs   SPOOF (SYNTHETIC / CONVERTED SPEECH)
```

The system is evaluated across:
1. **Canonical ASVspoof 2019 Logical Access (`LA`)**: 121,461 audio utterances covering 19 distinct attack systems (`A01`–`A19`).
2. **Dual-Stream Light CNN (MFM-LCNN)**: Dual encoders processing high-resolution 2D LFCC (60 linear filterbanks) and 2D Log-Mel (80 perceptual filterbanks) with Max-Feature-Map non-linearities.
3. **Acoustic Hardening**: Dual-domain robustness combining time-domain noise, reverberation, and lowpass filtering with 2D SpecAugment (frequency and time masking) to emulate in-the-wild acoustic variability.
4. **Stress-Testing & Generalization**: Six acoustic degradation conditions (SNR 0 dB to 20 dB, room impulse response reverberation, clipping distortion).

---

## 2. Project Directory Tree

```text
WildSpoof_Project/
├── data/
│   ├── manifests/             # Manifests: train_la.csv, dev_la.csv, eval_la.csv, train.csv, test.csv
│   └── processed/             # Cached feature matrices
│
├── notebooks/                 # Sequential academic Jupyter Notebooks (01 to 05)
│
├── src/                       # Core modular package
│   ├── la_models.py           # DualStreamLCNN, LCNNEncoder, MFM, DeepSpoofCNN, ResNetSpoofNet
│   ├── la_dataset.py          # Fast audio loader, torchaudio LFCC & Mel, SpecAugment
│   ├── augmentation.py        # Additive noise, reverberation, lowpass filtering
│   ├── evaluate.py            # Accuracy, Precision, Recall, F1, ROC-AUC, EER computation
│   ├── config.py              # Reproducible path and hyperparameter configurations
│   └── utils.py               # Seed setting & reproducibility utilities
│
├── scripts/                   # Executable workflow pipelines
│   ├── prepare_la_manifests.py       # ASVspoof 2019 LA protocol parser & manifest generator
│   ├── train_la_system.py            # DualStreamLCNN baseline & augmented training
│   ├── evaluate_la_attacks.py        # Granular per-attack evaluation across A07-A19
│   ├── evaluate_la_robustness.py     # Acoustic degradation stress-testing pipeline
│   ├── cross_dataset_evaluation.py   # Zero-shot cross-corpus generalization
│   ├── run_experiments.py            # Baseline experiments on for-2seconds
│   ├── run_robustness.py             # Robustness experiments on for-2seconds
│   ├── extract_features.py           # Feature extraction for for-2seconds
│   ├── create_manifest.py            # Manifest generator for for-2seconds
│   ├── analyze_dataset.py            # Audio metadata scanning & distributions
│   └── generate_notebooks.py         # Notebook regenerator
│
├── models/                    # Trained model checkpoints
│   ├── la_academic_lcnn_baseline.pt  # Clean Dual-Stream LCNN (0.30% Val EER, 0.9997 Val AUC)
│   └── la_academic_lcnn_augmented.pt # In-the-Wild Hardened LCNN (+0.1953 AUC lift under 5 dB noise)
│
├── results/
│   ├── figures/               # Publication-ready plots, ROC curves, bar charts
│   ├── metrics/               # Evaluation CSV tables (overall, attack breakdown, robustness)
│   └── logs/                  # Training history logs
│
├── requirements.txt           # Python dependencies
└── README.md                  # Package documentation
```

---

## 3. Key Results Summary

### 3.1. ASVspoof 2019 LA Benchmark Performance

| Metric | Clean Baseline LCNN | Acoustically Augmented LCNN |
| :--- | :--- | :--- |
| **Validation Accuracy** | **99.50%** | **89.63%** |
| **Validation EER (%)** | **0.30%** | **8.63%** |
| **Optimal Operating Threshold** | 0.7039 | 0.2378 |

### 3.2. Unseen Attack Highlights (A07–A19)
- **A07 (Vocoder + GAN)**: **0.00% EER, 1.0000 AUC**
- **A08 (Neural Waveform)**: **0.00% EER, 1.0000 AUC**
- **A09 (Vocoder)**: **0.40% EER, 0.9999 AUC**
- **A11 (Griffin-Lim)**: **2.47% EER, 0.9963 AUC**
- **A16 (Waveform Concatenation)**: **0.87% EER, 0.9998 AUC**
- **A19 (Spectral Filtering)**: **0.73% EER, 0.9998 AUC**

### 3.3. Acoustic Robustness Lifts (Acoustic Hardening)
- **Noise 10 dB**: Augmented LCNN delivers **+0.1740 AUC lift** and cuts EER by **+11.60%**.
- **Noise 5 dB**: Augmented LCNN delivers **+0.1953 AUC lift** and cuts EER by **+15.20%**.
- **Reverberation**: Augmented LCNN achieves **0.8855 AUC** (**+0.1620 AUC lift**) and cuts EER by **+12.80%**.
- **Noise + Reverberation**: Augmented LCNN delivers **+0.1406 AUC lift** and cuts EER by **+10.80%**.

---

## 4. How to Run

```powershell
# 1. Prepare LA manifests
python scripts/prepare_la_manifests.py

# 2. Train Academic DualStreamLCNN
python scripts/train_la_system.py

# 3. Evaluate Unseen Attacks (A07-A19)
python scripts/evaluate_la_attacks.py

# 4. Stress-Test Acoustic Robustness
python scripts/evaluate_la_robustness.py

# 5. Evaluate Cross-Dataset Generalization
python scripts/cross_dataset_evaluation.py
```
