import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config, utils, augmentation, evaluate
from src.la_dataset import load_and_standardize_audio, _LFCC_EXTRACTOR, _MEL_EXTRACTOR
from src.la_models import DualStreamLCNN, DeepSpoofCNN

# Controlled Acoustic Stress Conditions (never mislabeled as 'unseen')
CONDITIONS = [
    ("Clean", "clean"),
    ("Noise 20 dB", "noise_20db"),
    ("Noise 10 dB", "noise_10db"),
    ("Noise 5 dB", "noise_5db"),
    ("Reverberation", "reverberation"),
    ("Noise + Reverberation", "noise_reverb"),
    ("Low-pass Filter (3.4 kHz)", "lowpass_filter")
]

def load_robustness_model(path, device):
    state_dict = torch.load(path, map_location=device)
    if "lfcc_branch.conv1.weight" in state_dict:
        model = DualStreamLCNN.load_from_checkpoint(path, device=device)
        feat_type = "dual"
    else:
        model = DeepSpoofCNN(in_channels=1).to(device)
        model.load_state_dict(state_dict)
        feat_type = "mel"
    model.eval()
    return model, feat_type

def evaluate_model_on_condition(model, feat_type, device, df_test, condition_key):
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for _, row in df_test.iterrows():
            filepath = row["filepath"]
            label = int(row["label"])

            # 1. Load and standardize audio (exactly 32,000 samples)
            audio = load_and_standardize_audio(filepath)
            # 2. Apply controlled acoustic degradation
            corrupted = augmentation.corrupt_audio(audio, condition=condition_key)
            w_t = torch.from_numpy(corrupted.astype(np.float32)).unsqueeze(0)

            if feat_type == "dual":
                lfcc = _LFCC_EXTRACTOR(w_t).to(device)
                mel = _MEL_EXTRACTOR(w_t).to(device)
                logit = model(lfcc, mel)
            else:
                mel = _MEL_EXTRACTOR(w_t).to(device)
                logit = model(mel)

            # Support both 2-class softmax logits and 1-logit sigmoid
            if logit.dim() > 0 and logit.shape[-1] == 2:
                score = torch.softmax(logit, dim=-1)[..., 1].item()
            else:
                score = torch.sigmoid(logit.squeeze(-1)).item()

            all_scores.append(score)
            all_labels.append(label)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = float(auc(fpr, tpr))
    eer, eer_thresh = evaluate.compute_eer(all_labels, all_scores)

    preds = (all_scores >= 0.5).astype(int)
    cm_info = evaluate.compute_confusion_matrix_breakdown(all_labels, preds)

    return {
        "Accuracy": cm_info["Accuracy"],
        "Precision": cm_info["Precision"],
        "Recall": cm_info["Recall"],
        "F1": cm_info["F1"],
        "ROC-AUC": roc_auc,
        "EER": eer,
        "EER (%)": eer * 100.0,
        "Threshold": eer_thresh
    }

def main(n_sample_per_class=250):
    utils.set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_manifest = config.MANIFESTS_DIR / "eval_la.csv"
    if not eval_manifest.exists():
        raise FileNotFoundError(f"Evaluation manifest missing: {eval_manifest}")

    df_eval = pd.read_csv(eval_manifest)

    # Paired test subset: identical audio utterances evaluated across both models
    bona_sample = df_eval[df_eval["label"] == 0].sample(n=min(n_sample_per_class, len(df_eval[df_eval["label"] == 0])),
                                                        random_state=config.RANDOM_SEED)
    spoof_sample = df_eval[df_eval["label"] == 1].sample(n=min(n_sample_per_class, len(df_eval[df_eval["label"] == 1])),
                                                         random_state=config.RANDOM_SEED)
    test_subset = pd.concat([bona_sample, spoof_sample]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    print(f"Paired Robustness Stress-Test Subset: {len(test_subset)} utterances "
          f"(Bona Fide: {(test_subset['label'] == 0).sum()}, Spoof: {(test_subset['label'] == 1).sum()})")

    models_to_test = [
        ("Baseline CNN", config.MODELS_DIR / "la_academic_lcnn_baseline.pt", config.MODELS_DIR / "la_cnn_baseline_mel.pt"),
        ("Augmented CNN", config.MODELS_DIR / "la_academic_lcnn_augmented.pt", config.MODELS_DIR / "la_cnn_augmented_mel.pt")
    ]

    loaded_models = []
    for model_name, path_primary, path_compat in models_to_test:
        chosen_path = path_primary if path_primary.exists() else path_compat
        if chosen_path.exists():
            m, feat_type = load_robustness_model(chosen_path, device)
            loaded_models.append((model_name, m, feat_type, chosen_path.name))
        else:
            print(f"Skipping {model_name} (checkpoint not found: {chosen_path})")

    if not loaded_models:
        print("No trained models found to evaluate for robustness.")
        return

    results = []

    for cond_name, cond_key in CONDITIONS:
        print(f"\n--- Stress-Testing Condition: {cond_name} ---")
        row = {"Condition": cond_name}
        for model_name, model, feat_type, model_fname in loaded_models:
            print(f"  Evaluating {model_name} ({model_fname})...")
            metrics = evaluate_model_on_condition(model, feat_type, device, test_subset, cond_key)
            row[f"{model_name} AUC"] = metrics["ROC-AUC"]
            row[f"{model_name} EER (%)"] = metrics["EER (%)"]
            row[f"{model_name} F1"] = metrics["F1"]

        if len(loaded_models) == 2:
            row["AUC Lift"] = row["Augmented CNN AUC"] - row["Baseline CNN AUC"]
            row["EER Reduction (%)"] = row["Baseline CNN EER (%)"] - row["Augmented CNN EER (%)"]

        results.append(row)

    res_df = pd.DataFrame(results)
    print("\n=== ASVspoof 2019 LA Robustness Stress-Test Results ===")
    print(res_df.to_string(index=False))

    out_csv = config.METRICS_DIR / "la_robustness_comparison.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

    # Plot Robustness Comparison
    cond_labels = [c[0] for c in CONDITIONS]
    plt.figure(figsize=(11, 5.5), dpi=300)

    if "Baseline CNN AUC" in res_df.columns and "Augmented CNN AUC" in res_df.columns:
        plt.plot(cond_labels, res_df["Baseline CNN AUC"], marker='o', lw=2.5, color='#2980B9', label="Clean Baseline Dual LCNN")
        plt.plot(cond_labels, res_df["Augmented CNN AUC"], marker='s', lw=2.5, color='#C0392B', label="Acoustically Hardened Dual LCNN")
    elif "Baseline CNN AUC" in res_df.columns:
        plt.plot(cond_labels, res_df["Baseline CNN AUC"], marker='o', lw=2.5, color='#2980B9', label="Clean Baseline Dual LCNN")

    plt.title("Acoustic Robustness: Controlled Stress Conditions (ASVspoof 2019 LA)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Acoustic Stress Condition", fontsize=11, fontweight='bold')
    plt.ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    plt.ylim(0.40, 1.02)
    plt.xticks(rotation=15, ha='right', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_fig = config.FIGURES_DIR / "la_robustness_comparison.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved robustness comparison figure to {out_fig}")

    # Also save to 08_acoustic_robustness_stress_testing.png for report synchronization
    report_fig = config.FIGURES_DIR / "08_acoustic_robustness_stress_testing.png"
    plt.figure(figsize=(11, 5.5), dpi=300)
    plt.plot(cond_labels, res_df["Baseline CNN AUC"], marker='o', lw=2.5, color='#2980B9', label="Clean Baseline Dual LCNN")
    plt.plot(cond_labels, res_df["Augmented CNN AUC"], marker='s', lw=2.5, color='#C0392B', label="Acoustically Hardened Dual LCNN")
    plt.title("Acoustic Robustness: Controlled Stress Conditions (ASVspoof 2019 LA)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Acoustic Stress Condition", fontsize=11, fontweight='bold')
    plt.ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    plt.ylim(0.40, 1.02)
    plt.xticks(rotation=15, ha='right', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(report_fig, dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
