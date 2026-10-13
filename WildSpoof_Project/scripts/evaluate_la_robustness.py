import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config, utils, augmentation
from src.la_dataset import load_and_standardize_audio, extract_2d_mel, _LFCC_TRANSFORM, _MEL_TRANSFORM, _AMP_TO_DB
from src.la_models import DualStreamLCNN, DeepSpoofCNN
from src.evaluate import compute_eer

CONDITIONS = [
    ("Clean", "clean"),
    ("Noise 20 dB", "noise_20db"),
    ("Noise 10 dB", "noise_10db"),
    ("Noise 5 dB", "noise_5db"),
    ("Reverberation", "reverberation"),
    ("Noise + Reverberation", "noise_reverb")
]

def load_robustness_model(path, device):
    state_dict = torch.load(path, map_location=device)
    if "lfcc_branch.conv1.weight" in state_dict:
        model = DualStreamLCNN().to(device)
        model.load_state_dict(state_dict)
        feat_type = "dual"
    else:
        model = DeepSpoofCNN(in_channels=1).to(device)
        model.load_state_dict(state_dict)
        feat_type = "mel"
    return model, feat_type

def evaluate_model_on_condition(model, feat_type, device, df_test, condition_key):
    model.eval()
    all_scores, all_labels = [], []

    with torch.no_grad():
        for _, row in df_test.iterrows():
            filepath = row["filepath"]
            label = int(row["label"])

            # Load and standardize audio
            audio = load_and_standardize_audio(filepath)
            # Apply degradation
            corrupted = augmentation.corrupt_audio(audio, condition=condition_key)
            w_t = torch.from_numpy(corrupted.astype(np.float32)).unsqueeze(0)


            if feat_type == "dual":
                lfcc = _LFCC_TRANSFORM(w_t).to(device)
                mel = _AMP_TO_DB(_MEL_TRANSFORM(w_t)).to(device)
                logit = model(lfcc, mel)
            else:
                mel = extract_2d_mel(corrupted)
                mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
                logit = model(mel_tensor)

            score = torch.sigmoid(logit).item()
            all_scores.append(score)
            all_labels.append(label)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    try:
        eer, _ = compute_eer(all_labels, all_scores)
    except Exception:
        eer = 0.5
    preds = (all_scores >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary", zero_division=0)

    return {
        "Accuracy": acc,
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "EER (%)": eer * 100
    }

def main():
    utils.set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_manifest = config.MANIFESTS_DIR / "eval_la.csv"
    df_eval = pd.read_csv(eval_manifest)

    # Sample representative subset for robustness stress test: 250 bona fide + 250 spoof
    n_sample = 250
    bona_sample = df_eval[df_eval["label"] == 0].sample(n=n_sample, random_state=config.RANDOM_SEED)
    spoof_sample = df_eval[df_eval["label"] == 1].sample(n=n_sample, random_state=config.RANDOM_SEED)
    test_subset = pd.concat([bona_sample, spoof_sample]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    print(f"Robustness Test Subset: {len(test_subset)} utterances (Bona fide: 250, Spoof: 250)")

    # Load models
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
    print(f"Results saved to {out_csv}")

    # Plot Robustness Comparison
    cond_labels = [c[0] for c in CONDITIONS]
    plt.figure(figsize=(10, 5))

    if "Baseline CNN AUC" in res_df.columns and "Augmented CNN AUC" in res_df.columns:
        plt.plot(cond_labels, res_df["Baseline CNN AUC"], marker='o', lw=2.5, color='#4A90E2', label="Baseline CNN")
        plt.plot(cond_labels, res_df["Augmented CNN AUC"], marker='s', lw=2.5, color='#E94E77', label="Augmented CNN (Robust)")
    elif "Baseline CNN AUC" in res_df.columns:
        plt.plot(cond_labels, res_df["Baseline CNN AUC"], marker='o', lw=2.5, color='#4A90E2', label="Baseline CNN")

    plt.title("Acoustic Robustness Degradation on ASVspoof 2019 LA", fontsize=13, fontweight='bold')
    plt.xlabel("Acoustic Degradation Condition", fontsize=11, fontweight='bold')
    plt.ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    plt.ylim(0.45, 1.02)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_fig = config.FIGURES_DIR / "la_robustness_comparison.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved robustness comparison plot to {out_fig}")

if __name__ == "__main__":
    main()
