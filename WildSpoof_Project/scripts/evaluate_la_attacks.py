import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config, utils, evaluate
from src.la_dataset import ASVspoofLADataset
from src.la_models import DualStreamLCNN, DeepSpoofCNN

def load_eval_model(model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    if "lfcc_branch.conv1.weight" in state_dict:
        model = DualStreamLCNN.load_from_checkpoint(model_path, device=device)
        feature_type = "dual"
    else:
        model = DeepSpoofCNN(in_channels=1).to(device)
        model.load_state_dict(state_dict)
        feature_type = "mel"
    model.eval()
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

            if logits.dim() > 1 and logits.shape[1] == 2:
                scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            else:
                scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
            all_audio_ids.extend(audio_ids)
            all_system_ids.extend(system_ids)

    return np.array(all_scores), np.array(all_labels), all_audio_ids, all_system_ids

def evaluate_subset(labels, scores, name="Subset"):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return {"Subset": name, "Samples": len(labels), "ROC-AUC": np.nan, "EER": np.nan, "EER (%)": np.nan,
                "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan}

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))
    eer, thresh = evaluate.compute_eer(labels, scores)

    preds = (scores >= 0.5).astype(int)
    cm_info = evaluate.compute_confusion_matrix_breakdown(labels, preds)

    return {
        "Subset": name,
        "Samples": len(labels),
        "Accuracy": cm_info["Accuracy"],
        "Precision": cm_info["Precision"],
        "Recall": cm_info["Recall"],
        "F1": cm_info["F1"],
        "ROC-AUC": roc_auc,
        "EER": eer,
        "EER (%)": eer * 100.0,
        "Threshold": thresh
    }

def main(n_eval_bona=1500, n_per_attack=115):
    utils.set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_manifest = config.MANIFESTS_DIR / "eval_la.csv"
    if not eval_manifest.exists():
        raise FileNotFoundError(f"Evaluation manifest missing: {eval_manifest}")

    df_eval = pd.read_csv(eval_manifest)

    # Stratified balanced evaluation set: 1,500 bona fide + 115 per attack across A07-A19 (~1,495 spoof)
    bona_eval = df_eval[df_eval["label"] == 0].sample(n=min(n_eval_bona, len(df_eval[df_eval["label"] == 0])),
                                                      random_state=config.RANDOM_SEED)

    eval_attacks = sorted(df_eval[df_eval["label"] == 1]["system_id"].unique())
    eval_spoof_parts = []
    for att in eval_attacks:
        sub = df_eval[df_eval["system_id"] == att]
        eval_spoof_parts.append(sub.sample(n=min(n_per_attack, len(sub)), random_state=config.RANDOM_SEED))
    spoof_eval = pd.concat(eval_spoof_parts)

    df_eval_sampled = pd.concat([bona_eval, spoof_eval]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    print(f"ASVspoof 2019 LA Evaluation Utterances: {len(df_eval_sampled)} "
          f"(Bona Fide: {(df_eval_sampled['label'] == 0).sum()}, Spoof: {(df_eval_sampled['label'] == 1).sum()})")

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
    print(f"Saved overall metrics to {overall_csv}")

    # 2. Granular Attack Breakdown (A07 to A19 Unseen Attacks)
    attack_rows = []
    labels_eval = df_eval_sampled["label"].values
    bona_mask = (labels_eval == 0)

    for att in eval_attacks:
        att_mask = (df_eval_sampled["system_id"].values == att)
        curr_mask = bona_mask | att_mask
        curr_labels = labels_eval[curr_mask]

        meta = df_eval_sampled[att_mask].iloc[0]
        row = {
            "Attack ID": att,
            "Samples": int(att_mask.sum()),
            "Type": meta["attack_type"],
            "Method": meta["attack_method"]
        }

        for model_name, data in model_preds.items():
            curr_scores = data["scores"][curr_mask]
            sub_res = evaluate_subset(curr_labels, curr_scores, name=att)
            row[f"{model_name} AUC"] = sub_res["ROC-AUC"]
            row[f"{model_name} EER (%)"] = sub_res["EER (%)"]

        if "Baseline CNN" in model_preds and "Augmented CNN" in model_preds:
            row["Delta AUC"] = row["Augmented CNN AUC"] - row["Baseline CNN AUC"]
            row["Delta EER (%)"] = row["Baseline CNN EER (%)"] - row["Augmented CNN EER (%)"]

        attack_rows.append(row)

    attack_df = pd.DataFrame(attack_rows)
    print("\n=== Granular Attack Breakdown (A07 - A19 Unseen Attacks) ===")
    print(attack_df.to_string(index=False))

    # Identify easiest and hardest attacks automatically
    if "Baseline CNN EER (%)" in attack_df.columns:
        easiest = attack_df.loc[attack_df["Baseline CNN EER (%)"].idxmin()]
        hardest = attack_df.loc[attack_df["Baseline CNN EER (%)"].idxmax()]
        print(f"\n[Automated Attack Analysis]")
        print(f"  Easiest Attack for Baseline: {easiest['Attack ID']} ({easiest['Type']} - {easiest['Method']}) with EER: {easiest['Baseline CNN EER (%)']:.2f}%")
        print(f"  Hardest Attack for Baseline: {hardest['Attack ID']} ({hardest['Type']} - {hardest['Method']}) with EER: {hardest['Baseline CNN EER (%)']:.2f}%")

    attack_csv = config.METRICS_DIR / "la_attack_breakdown.csv"
    attack_df.to_csv(attack_csv, index=False)
    print(f"Saved attack breakdown to {attack_csv}")

    # 3. Visualization: Per-Attack EER Bar Chart
    plt.figure(figsize=(12, 5.5), dpi=300)
    x = np.arange(len(eval_attacks))
    width = 0.35

    if "Baseline CNN" in model_preds and "Augmented CNN" in model_preds:
        plt.bar(x - width/2, attack_df["Baseline CNN EER (%)"], width, label="Clean Baseline Dual LCNN", color="#2980B9")
        plt.bar(x + width/2, attack_df["Augmented CNN EER (%)"], width, label="Acoustically Hardened Dual LCNN", color="#C0392B")
    elif "Baseline CNN" in model_preds:
        plt.bar(x, attack_df["Baseline CNN EER (%)"], width, label="Clean Baseline Dual LCNN", color="#2980B9")

    plt.xlabel("ASVspoof 2019 Unseen Attack ID (A07 - A19)", fontsize=11, fontweight='bold')
    plt.ylabel("Equal Error Rate (EER %)", fontsize=11, fontweight='bold')
    plt.title("Granular Breakdown: Unseen Attacks (A07–A19) on ASVspoof 2019 LA", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, [f"{a}\n({attack_df.loc[i, 'Type']})" for i, a in enumerate(eval_attacks)], rotation=0, fontsize=9, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    out_eer_fig = config.FIGURES_DIR / "06_unseen_attacks_eer_barchart.png"
    plt.savefig(out_eer_fig, dpi=300)
    plt.close()
    print(f"Saved attack EER bar chart to {out_eer_fig}")

    # 4. Visualization: Per-Attack AUC Bar Chart
    plt.figure(figsize=(12, 5.5), dpi=300)
    if "Baseline CNN" in model_preds and "Augmented CNN" in model_preds:
        plt.bar(x - width/2, attack_df["Baseline CNN AUC"], width, label="Clean Baseline Dual LCNN", color="#2980B9")
        plt.bar(x + width/2, attack_df["Augmented CNN AUC"], width, label="Acoustically Hardened Dual LCNN", color="#C0392B")
    plt.axhline(0.5, color='gray', linestyle='--', label="Chance Level (0.50)")
    plt.ylim(0.5, 1.05)
    plt.xlabel("ASVspoof 2019 Unseen Attack ID (A07 - A19)", fontsize=11, fontweight='bold')
    plt.ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    plt.title("Granular Discriminability: Unseen Attacks (A07–A19) on ASVspoof 2019 LA", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, [f"{a}\n({attack_df.loc[i, 'Type']})" for i, a in enumerate(eval_attacks)], rotation=0, fontsize=9, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    out_auc_fig = config.FIGURES_DIR / "07_unseen_attacks_auc_barchart.png"
    plt.savefig(out_auc_fig, dpi=300)
    plt.close()
    print(f"Saved attack AUC bar chart to {out_auc_fig}")

    # 5. Combined ROC Curves
    plt.figure(figsize=(7.5, 6.5), dpi=300)
    colors = {"Baseline CNN": "#2980B9", "Augmented CNN": "#C0392B"}
    for model_name, data in model_preds.items():
        fpr, tpr, _ = roc_curve(data["labels"], data["scores"])
        roc_auc = auc(fpr, tpr)
        eer_val, _ = evaluate.compute_eer(data["labels"], data["scores"])
        c = colors.get(model_name, "#27AE60")
        plt.plot(fpr, tpr, lw=2.5, color=c, label=f"{model_name} (AUC = {roc_auc:.4f}, EER = {eer_val*100:.2f}%)")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (Bona Fide False Alarm)", fontsize=11, fontweight='bold')
    plt.ylabel("True Positive Rate (Spoof Detection Rate)", fontsize=11, fontweight='bold')
    plt.title("Official ASVspoof 2019 LA Evaluation ROC Curves\n(Unseen Attack Systems A07–A19)", fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    out_roc_fig = config.FIGURES_DIR / "05_roc_curves_official_la.png"
    plt.savefig(out_roc_fig, dpi=300)
    plt.close()
    print(f"Saved ROC curves to {out_roc_fig}")

    # 6. Baseline Confusion Matrix Figure directly from raw predictions
    if "Baseline CNN" in model_preds:
        base_data = model_preds["Baseline CNN"]
        preds_base = (base_data["scores"] >= 0.5).astype(int)
        cm_info_base = evaluate.compute_confusion_matrix_breakdown(base_data["labels"], preds_base)

        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        cm_norm = cm_info_base["CM_row_norm_pct"]
        cm_raw = cm_info_base["CM_raw"]
        sns.heatmap(cm_norm, annot=False, cmap='Blues', vmin=0, vmax=100, ax=ax,
                    xticklabels=['Bona Fide (0)', 'Spoof (1)'],
                    yticklabels=['Bona Fide (0)', 'Spoof (1)'])

        labels = [
            [f"TN: {cm_raw[0, 0]}\n({cm_norm[0, 0]:.1f}%)", f"FP: {cm_raw[0, 1]}\n({cm_norm[0, 1]:.1f}%)"],
            [f"FN: {cm_raw[1, 0]}\n({cm_norm[1, 0]:.1f}%)", f"TP: {cm_raw[1, 1]}\n({cm_norm[1, 1]:.1f}%)"]
        ]
        for i in range(2):
            for j in range(2):
                text_color = "white" if cm_norm[i, j] > 50 else "black"
                ax.text(j + 0.5, i + 0.5, labels[i][j], ha="center", va="center",
                        color=text_color, fontsize=11, fontweight="bold")

        ax.set_title("Confusion Matrix: Official ASVspoof 2019 LA Evaluation\n(Dual-Stream MFM-LCNN Baseline — Row-Normalized %)",
                     fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
        ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
        plt.tight_layout()

        cm_fig_path = config.FIGURES_DIR / "04_confusion_matrix_evaluation.png"
        plt.savefig(cm_fig_path, dpi=300)
        plt.close()
        print(f"Saved verified confusion matrix figure to {cm_fig_path}")

if __name__ == "__main__":
    main()
