# Overleaf Springer LNCS Report Instructions

This directory contains the ready-to-compile LaTeX source code for the project report formatted according to the **Springer Conference Proceedings (LNCS)** template:
[Springer Conference Proceedings Template on Overleaf (Updated 2022-01-12)](https://www.overleaf.com/latex/templates/springer-conference-proceedings-template-updated-2022-01-12/wcvbtmwtykqj)

---

## Files in this Directory

- `main.tex`: The primary LaTeX file using `\documentclass[runningheads]{llncs}`, complete with all sections, mathematical equations, listing blocks, and figures.
- `references.bib`: The BibTeX bibliography file containing all 15 academic references with full DOI links.
- `figures/`: Subfolder containing all high-resolution figures (`.png`) referenced in `main.tex`:
  - `01_dataset_architecture_and_splits.png`
  - `02_methodology_flowchart.png`
  - `03_model_evaluation_metrics_card.png`
  - `04_confusion_matrix_evaluation.png`
  - `05_roc_curves_official_la.png`
  - `06_unseen_attacks_eer_barchart.png`
  - `07_unseen_attacks_auc_barchart.png`
  - `08_acoustic_robustness_stress_testing.png`
  - `09_feature_representations_comparison.png`
  - `10_training_convergence_curves.png`
  - `11_sota_model_benchmark_comparison.png`
- `overleaf_springer_project.zip`: Pre-packaged ready-to-upload ZIP archive for Overleaf.

---

## How to Open in Overleaf

### Method 1: Using the Pre-made ZIP (Fastest & Recommended)
1. Open [Overleaf](https://www.overleaf.com/).
2. On your dashboard, click **"New Project"** $\rightarrow$ **"Upload Project"**.
3. Select or drag-and-drop the generated ZIP file:
   `d:\Speech\PROJECT\reports\overleaf_springer_project.zip`
4. If Overleaf prompts that `llncs.cls` is needed, you can either:
   - Select the Springer Conference Proceedings template and drop `main.tex` and `figures/` inside, OR
   - Download `llncs.cls` from the template link and upload it into the project root.
5. Click **Recompile** in Overleaf.

### Method 2: Using the Template Link Directly
1. Navigate to the template link:  
   [https://www.overleaf.com/latex/templates/springer-conference-proceedings-template-updated-2022-01-12/wcvbtmwtykqj](https://www.overleaf.com/latex/templates/springer-conference-proceedings-template-updated-2022-01-12/wcvbtmwtykqj)
2. Click **"Open as Template"**.
3. In the left-hand file tree in Overleaf, click the upload button to upload the `figures` folder (or upload `overleaf_springer_project.zip`).
4. Replace the contents of `samplepaper.tex` (or `main.tex`) with the code from `reports/main.tex`.
5. Upload `references.bib`.
6. Click **Recompile** (Ctrl+Enter or green "Recompile" button).

---

## Key Features of this Report
- **Humanized Academic Tone**: Written in authentic researcher voice with signal processing physics, clear mathematical notation, and candid empirical discussions.
- **Full Coverage of ASVspoof 2019 LA**: Strict speaker-disjoint splits (121,461 utterances), known attacks A01--A06 vs. unseen zero-day attacks A07--A19.
- **Mathematical Formulations**: Linear Source-Filter model, STFT windowing & sidelobe suppression, LFCC filterbank mechanics vs. Mel compression, MFM channel-splitting non-linearities, and class-weighted BCE loss.
- **In-Depth Benchmark Analysis**: Rigorous deconstruction of the 0.9032 ROC-AUC metric—explaining why near-zero errors on vocoders/GANs and challenges on A12/A13 make 0.9032 a competitive academic result.
- **Stress-Testing Tables & SOTA Comparisons**: Detailed tables showing acoustic hardening (+0.1953 AUC lift under 5 dB noise) and SOTA positioning against GMMs, RawNet2, and AASIST.
