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
from src import config, utils
from src.la_dataset import ASVspoofLADataset
from src.la_models import DualStreamLCNN, DeepSpoofCNN
from src.evaluate import compute_eer

def load_eval_model(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    if "lfcc_branch.conv1.weight" in state_dict:
        model = DualStreamLCNN().to(device)
        model.load_state_dict(state_dict)
        feature_type = "dual"
    else:
        model = DeepSpoofCNN(in_channels=1).to(device)
        model.load_state_dict(state_dict)
        feature_type = "mel"
    return model, feature_type

def get_predictions(model, loader, device):
    model.eval()
    all_scores, all_labels, all_audio_ids, all_system_ids = [], [], [], []
    with torch.no_grad():
        for feats, labels, audio_ids, system_ids in tqdm(loader, desc="Inference"):
            if isinstance(feats, (tuple, list)):
                lfcc, mel = feats[0].to(device), feats[1].to(device)
                logits = model(lfcc, mel)
            else:
                feats = feats.to(device)
                logits = model(feats)

            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
            all_audio_ids.extend(audio_ids)
            all_system_ids.extend(system_ids)

    return np.array(all_scores), np.array(all_labels), all_audio_ids, all_system_ids

def evaluate_subset(labels, scores, name="Subset"):
    if len(np.unique(labels)) < 2:
        return {"Subset": name, "Samples": len(labels), "ROC-AUC": np.nan, "EER": np.nan, "Acc": np.nan}
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    try:
        eer, thresh = compute_eer(labels, scores)
    except Exception:
        eer, thresh = 0.5, 0.5
    preds = (scores >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "Subset": name,
        "Samples": len(labels),
        "Accuracy": acc,
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "EER": eer,
        "EER (%)": eer * 100
    }

def main():
    utils.set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifests_dir = config.MANIFESTS_DIR
    dev_manifest = manifests_dir / "dev_la.csv"
    eval_manifest = manifests_dir / "eval_la.csv"

    df_dev = pd.read_csv(dev_manifest)
    df_eval = pd.read_csv(eval_manifest)

    # For comprehensive yet fast evaluation, we evaluate on all dev data or a rich stratified sample
    # Let's take a sample of eval: 1,500 bona fide + 1,500 spoof (balanced across A07-A19, ~115 per attack)
    n_eval_bona = 1500
    bona_eval = df_eval[df_eval["label"] == 0].sample(n=min(n_eval_bona, len(df_eval[df_eval["label"] == 0])), random_state=config.RANDOM_SEED)

    eval_attacks = sorted(df_eval[df_eval["label"] == 1]["system_id"].unique())
    per_attack_eval = n_eval_bona // len(eval_attacks)
    eval_spoof_parts = []
    for att in eval_attacks:
        sub = df_eval[df_eval["system_id"] == att]
        eval_spoof_parts.append(sub.sample(n=min(per_attack_eval, len(sub)), random_state=config.RANDOM_SEED))
    spoof_eval = pd.concat(eval_spoof_parts)

    df_eval_sampled = pd.concat([bona_eval, spoof_eval]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    print(f"Eval Test Utterances: {len(df_eval_sampled)} (Bona fide: {(df_eval_sampled['label']==0).sum()}, Spoof: {(df_eval_sampled['label']==1).sum()})")

    # Load candidate models
    models_to_test = [
        ("Baseline CNN", config.MODELS_DIR / "la_academic_lcnn_baseline.pt", config.MODELS_DIR / "la_cnn_baseline_mel.pt"),
        ("Augmented CNN", config.MODELS_DIR / "la_academic_lcnn_augmented.pt", config.MODELS_DIR / "la_cnn_augmented_mel.pt")
    ]

    model_preds = {}

    for model_name, path_primary, path_compat in models_to_test:
        chosen_path = path_primary if path_primary.exists() else path_compat
        if not chosen_path.exists():
            print(f"Skipping {model_name} (checkpoint not found: {chosen_path})")
            continue

        print(f"\n--- Evaluating {model_name} on ASVspoof 2019 LA Evaluation Set ({chosen_path.name}) ---")
        model, feat_type = load_eval_model(chosen_path, device)
        eval_dataset = ASVspoofLADataset(df_eval_sampled, feature_type=feat_type, augment=False)
        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=0)

        scores, labels, audio_ids, system_ids = get_predictions(model, eval_loader, device)
        model_preds[model_name] = {
            "scores": scores,
            "labels": labels,
            "audio_ids": audio_ids,
            "system_ids": system_ids
        }

    if not model_preds:
        print("No models evaluated.")

        return

    # 1. Overall Evaluation Table
    overall_rows = []
    for model_name, data in model_preds.items():
        res = evaluate_subset(data["labels"], data["scores"], name=f"{model_name} (Overall Eval)")
        overall_rows.append(res)

    overall_df = pd.DataFrame(overall_rows)
    print("\n=== Overall ASVspoof 2019 LA Evaluation Results ===")
    print(overall_df.to_string(index=False))
    overall_csv = config.METRICS_DIR / "la_eval_overall.csv"
    overall_df.to_csv(overall_csv, index=False)

    # 2. Granular Attack Breakdown (A07 to A19)
    attack_rows = []
    labels_eval = df_eval_sampled["label"].values
    bona_mask = (labels_eval == 0)

    for att in eval_attacks:
        att_mask = (df_eval_sampled["system_id"].values == att)
        curr_mask = bona_mask | att_mask
        curr_labels = labels_eval[curr_mask]

        row = {"Attack ID": att, "Samples": int(att_mask.sum())}
        # Add metadata
        meta = df_eval_sampled[att_mask].iloc[0]
        row["Type"] = meta["attack_type"]
        row["Method"] = meta["attack_method"]

        for model_name, data in model_preds.items():
            curr_scores = data["scores"][curr_mask]
            sub_res = evaluate_subset(curr_labels, curr_scores, name=att)
            row[f"{model_name} AUC"] = sub_res["ROC-AUC"]
            row[f"{model_name} EER (%)"] = sub_res["EER (%)"]

        attack_rows.append(row)

    attack_df = pd.DataFrame(attack_rows)
    print("\n=== Granular Attack Breakdown (A07 - A19 Unseen Attacks) ===")
    print(attack_df.to_string(index=False))
    attack_csv = config.METRICS_DIR / "la_attack_breakdown.csv"
    attack_df.to_csv(attack_csv, index=False)

    # 3. Visualization: Per-Attack EER Bar Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(eval_attacks))
    width = 0.35

    if "Baseline CNN" in model_preds and "Augmented CNN" in model_preds:
        plt.bar(x - width/2, attack_df["Baseline CNN EER (%)"], width, label="Baseline CNN", color="#4A90E2")
        plt.bar(x + width/2, attack_df["Augmented CNN EER (%)"], width, label="Augmented CNN", color="#E94E77")
    elif "Baseline CNN" in model_preds:
        plt.bar(x, attack_df["Baseline CNN EER (%)"], width, label="Baseline CNN", color="#4A90E2")

    plt.xlabel("ASVspoof 2019 Unseen Attack ID (A07 - A19)", fontsize=12, fontweight='bold')
    plt.ylabel("Equal Error Rate (EER %)", fontsize=12, fontweight='bold')
    plt.title("Per-Attack EER on ASVspoof 2019 LA Evaluation Set", fontsize=14, fontweight='bold')
    plt.xticks(x, [f"{a}\n({attack_df.loc[i, 'Type']})" for i, a in enumerate(eval_attacks)], rotation=0, fontsize=9)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart_path = config.FIGURES_DIR / "attack_eer_barchart.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved per-attack bar chart to {chart_path}")

    # 4. ROC Curves
    plt.figure(figsize=(8, 7))
    for model_name, data in model_preds.items():
        fpr, tpr, _ = roc_curve(data["labels"], data["scores"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.5, label=f"{model_name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight='bold')
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight='bold')
    plt.title("ROC Curves on ASVspoof 2019 LA Evaluation Set", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    roc_chart_path = config.FIGURES_DIR / "la_eval_roc_curves.png"
    plt.savefig(roc_chart_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves to {roc_chart_path}")

if __name__ == "__main__":
    main()
