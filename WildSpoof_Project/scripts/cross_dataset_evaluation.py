import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config, utils, models
from src.la_models import DualStreamLCNN, DeepSpoofCNN
from src.la_dataset import load_and_standardize_audio, extract_2d_mel, _LFCC_TRANSFORM, _MEL_TRANSFORM, _AMP_TO_DB
from src.evaluate import compute_eer

def evaluate_on_dataset(model, feat_type, device, df, name="Dataset"):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            filepath = row["filepath"]
            label = int(row["label"])

            audio = load_and_standardize_audio(filepath, target_samples=32000)
            w_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)


            if feat_type == "dual":
                lfcc = _LFCC_TRANSFORM(w_t).to(device)
                mel = _AMP_TO_DB(_MEL_TRANSFORM(w_t)).to(device)
                logit = model(lfcc, mel)
            else:
                mel = extract_2d_mel(audio)
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

    return {
        "Test Dataset": name,
        "Samples": len(df),
        "Accuracy": acc,
        "ROC-AUC": roc_auc,
        "EER (%)": eer * 100
    }

def main():
    utils.set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load FoR test set
    for_test_csv = config.MANIFESTS_DIR / "test.csv"
    df_for_test = pd.read_csv(for_test_csv)
    # Subsample 500 samples
    df_for_sub = pd.concat([
        df_for_test[df_for_test["label"] == 0].sample(n=250, random_state=config.RANDOM_SEED),
        df_for_test[df_for_test["label"] == 1].sample(n=250, random_state=config.RANDOM_SEED)
    ]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # Load LA eval set
    la_eval_csv = config.MANIFESTS_DIR / "eval_la.csv"
    df_la_eval = pd.read_csv(la_eval_csv)
    # Subsample 500 samples
    df_la_sub = pd.concat([
        df_la_eval[df_la_eval["label"] == 0].sample(n=250, random_state=config.RANDOM_SEED),
        df_la_eval[df_la_eval["label"] == 1].sample(n=250, random_state=config.RANDOM_SEED)
    ]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    print(f"FoR Test Utterances: {len(df_for_sub)}")
    print(f"LA Test Utterances: {len(df_la_sub)}")

    matrix_rows = []

    # 1. Evaluate FoR Model (models/cnn_augmented.pt or models/best_cnn_model.pt)
    for_model_path = config.MODELS_DIR / "cnn_augmented.pt"
    if not for_model_path.exists():
        for_model_path = config.MODELS_DIR / "best_cnn_model.pt"

    if for_model_path.exists():
        print(f"\n--- Loading FoR Trained Model from {for_model_path.name} ---")
        m_for = models.SpoofCNN().to(device)
        m_for.load_state_dict(torch.load(for_model_path, map_location=device))

        res_within = evaluate_on_dataset(m_for, "mel", device, df_for_sub, name="FoR (Within-Domain)")
        res_cross = evaluate_on_dataset(m_for, "mel", device, df_la_sub, name="ASVspoof LA (Cross-Domain)")

        matrix_rows.append({
            "Trained On": "FoR Dataset",
            "Evaluated On": "FoR (Within)",
            "ROC-AUC": res_within["ROC-AUC"],
            "EER (%)": res_within["EER (%)"],
            "Accuracy": res_within["Accuracy"]
        })
        matrix_rows.append({
            "Trained On": "FoR Dataset",
            "Evaluated On": "ASVspoof LA (Cross)",
            "ROC-AUC": res_cross["ROC-AUC"],
            "EER (%)": res_cross["EER (%)"],
            "Accuracy": res_cross["Accuracy"]
        })

    # 2. Evaluate LA Model
    la_candidates = [
        config.MODELS_DIR / "la_academic_lcnn_augmented.pt",
        config.MODELS_DIR / "la_cnn_augmented_mel.pt",
        config.MODELS_DIR / "la_academic_lcnn_baseline.pt",
        config.MODELS_DIR / "la_cnn_baseline_mel.pt"
    ]
    la_model_path = None
    for p in la_candidates:
        if p.exists():
            la_model_path = p
            break

    if la_model_path:
        print(f"\n--- Loading LA Trained Model from {la_model_path.name} ---")
        sd = torch.load(la_model_path, map_location=device)
        if "lfcc_branch.conv1.weight" in sd:
            m_la = DualStreamLCNN().to(device)
            feat_type_la = "dual"
        else:
            m_la = DeepSpoofCNN(in_channels=1).to(device)
            feat_type_la = "mel"
        m_la.load_state_dict(sd)

        res_within_la = evaluate_on_dataset(m_la, feat_type_la, device, df_la_sub, name="ASVspoof LA (Within-Domain)")
        res_cross_la = evaluate_on_dataset(m_la, feat_type_la, device, df_for_sub, name="FoR (Cross-Domain)")

        matrix_rows.append({
            "Trained On": "ASVspoof LA",
            "Evaluated On": "ASVspoof LA (Within)",
            "ROC-AUC": res_within_la["ROC-AUC"],
            "EER (%)": res_within_la["EER (%)"],
            "Accuracy": res_within_la["Accuracy"]
        })
        matrix_rows.append({
            "Trained On": "ASVspoof LA",
            "Evaluated On": "FoR (Cross)",
            "ROC-AUC": res_cross_la["ROC-AUC"],
            "EER (%)": res_cross_la["EER (%)"],
            "Accuracy": res_cross_la["Accuracy"]
        })


    cross_df = pd.DataFrame(matrix_rows)
    print("\n=== Cross-Dataset Generalization Matrix ===")
    print(cross_df.to_string(index=False))

    out_csv = config.METRICS_DIR / "cross_dataset_generalization.csv"
    cross_df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")

    # Plot Matrix Bar Chart
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        [f"{r['Trained On']}\n-> {r['Evaluated On']}" for _, r in cross_df.iterrows()],
        cross_df["ROC-AUC"],
        color=['#2ECC71', '#E67E22', '#3498DB', '#9B59B6'],
        width=0.5
    )
    plt.axhline(0.5, color='gray', linestyle='--', label="Chance Level (0.50)")
    plt.ylim(0.0, 1.05)
    plt.ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    plt.title("Cross-Dataset Generalization: FoR vs. ASVspoof 2019 LA", fontsize=13, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    out_fig = config.FIGURES_DIR / "cross_dataset_generalization.png"
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved cross-dataset plot to {out_fig}")

if __name__ == "__main__":
    main()
