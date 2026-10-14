import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config, utils, augmentation, evaluate
from src.la_dataset import ASVspoofLADataset, load_and_standardize_audio, _LFCC_EXTRACTOR, _MEL_EXTRACTOR
from src.la_models import DualStreamLCNN, get_model_summary

def run_evaluation_pipeline(n_eval_bona=1500, n_per_attack=115, n_robustness_sample=250):
    """
    Unified end-to-end evaluation pipeline that produces 100% reproducible,
    empirically verified results directly from model checkpoints on disk.
    Never uses hard-coded numbers.
    """
    utils.set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"RUNNING RESEARCH-GRADE RESULT GENERATOR (Device: {device})")
    print(f"=======================================================")

    eval_manifest = config.MANIFESTS_DIR / "eval_la.csv"
    if not eval_manifest.exists():
        raise FileNotFoundError(f"Manifest missing: {eval_manifest}. Please run prepare_la_manifests.py.")

    df_eval = pd.read_csv(eval_manifest)

    # -------------------------------------------------------------
    # 1. Model Checkpoint Loading & Architecture Audit
    # -------------------------------------------------------------
    models_config = [
        ("Baseline CNN", config.MODELS_DIR / "la_academic_lcnn_baseline.pt"),
        ("Augmented CNN", config.MODELS_DIR / "la_academic_lcnn_augmented.pt")
    ]

    loaded_models = {}
    model_summaries = []

    for name, path in models_config:
        if not path.exists():
            raise FileNotFoundError(f"Required model checkpoint not found: {path}")

        m = DualStreamLCNN.load_from_checkpoint(path, device=device)
        loaded_models[name] = m
        summary = get_model_summary(m)
        file_size_bytes = os.path.getsize(path)

        model_summaries.append({
            "Model Name": name,
            "Checkpoint File": path.name,
            "Embedding Dim": m.embedding_dim,
            "Fused Dim": m.embedding_dim * 2,
            "Output Classes": m.num_classes,
            "Trainable Parameters": summary["trainable_parameters"],
            "Total Parameters": summary["total_parameters"],
            "FP32 Memory (KB)": round(summary["fp32_memory_kb"], 2),
            "FP16 Memory (KB)": round(summary["fp16_memory_kb"], 2),
            "Serialized File Size (Bytes)": file_size_bytes,
            "Serialized File Size (KB)": round(file_size_bytes / 1024.0, 2)
        })

    summary_df = pd.DataFrame(model_summaries)
    summary_csv = config.METRICS_DIR / "model_size_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("\n--- Model Architecture & Parameter Memory Summary ---")
    print(summary_df.to_string(index=False))

    # -------------------------------------------------------------
    # 2. Main Evaluation Set Stratification
    # -------------------------------------------------------------
    bona_eval = df_eval[df_eval["label"] == 0].sample(n=min(n_eval_bona, len(df_eval[df_eval["label"] == 0])),
                                                      random_state=config.RANDOM_SEED)

    eval_attacks = sorted(df_eval[df_eval["label"] == 1]["system_id"].unique())
    eval_spoof_parts = []
    for att in eval_attacks:
        sub = df_eval[df_eval["system_id"] == att]
        eval_spoof_parts.append(sub.sample(n=min(n_per_attack, len(sub)), random_state=config.RANDOM_SEED))
    spoof_eval = pd.concat(eval_spoof_parts)

    df_eval_sampled = pd.concat([bona_eval, spoof_eval]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print(f"\nEvaluated Sample Size: {len(df_eval_sampled)} utterances "
          f"(Bona Fide: {(df_eval_sampled['label'] == 0).sum()}, Spoof: {(df_eval_sampled['label'] == 1).sum()})")

    # -------------------------------------------------------------
    # 3. Model Inference on Evaluation Set
    # -------------------------------------------------------------
    eval_dataset = ASVspoofLADataset(df_eval_sampled, feature_type="dual", augment=False)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=0)

    model_preds = {}
    for name, model in loaded_models.items():
        print(f"Running inference for {name} on evaluation set...")
        all_scores, all_labels = [], []
        with torch.no_grad():
            for feats, labels, _, _ in tqdm(eval_loader, desc=f"Inference [{name}]"):
                lfcc, mel = feats[0].to(device), feats[1].to(device)
                logits = model(lfcc, mel)
                if logits.dim() > 1 and logits.shape[1] == 2:
                    scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                else:
                    scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

                all_scores.extend(scores)
                all_labels.extend(labels.numpy())

        model_preds[name] = {
            "scores": np.array(all_scores),
            "labels": np.array(all_labels)
        }

    # -------------------------------------------------------------
    # 4. Overall Evaluation Metrics & Sanity Checks
    # -------------------------------------------------------------
    overall_rows = []
    for name, data in model_preds.items():
        preds = (data["scores"] >= 0.5).astype(int)
        metrics = evaluate.evaluate_predictions(
            data["labels"], preds, data["scores"],
            model_name=name, feature_name="Dual LFCC+Mel", save_figures=False
        )
        overall_rows.append(metrics)

    overall_df = pd.DataFrame(overall_rows)
    overall_csv = config.METRICS_DIR / "la_eval_overall.csv"
    overall_df.to_csv(overall_csv, index=False)

    # -------------------------------------------------------------
    # 5. Granular Attack Breakdown (A07 - A19)
    # -------------------------------------------------------------
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

        for name, data in model_preds.items():
            curr_scores = data["scores"][curr_mask]
            fpr, tpr, _ = roc_curve(curr_labels, curr_scores)
            cur_auc = float(auc(fpr, tpr))
            cur_eer, _ = evaluate.compute_eer(curr_labels, curr_scores)
            row[f"{name} AUC"] = cur_auc
            row[f"{name} EER (%)"] = cur_eer * 100.0

        if "Baseline CNN" in model_preds and "Augmented CNN" in model_preds:
            row["Delta AUC"] = row["Augmented CNN AUC"] - row["Baseline CNN AUC"]
            row["Delta EER (%)"] = row["Baseline CNN EER (%)"] - row["Augmented CNN EER (%)"]

        attack_rows.append(row)

    attack_df = pd.DataFrame(attack_rows)
    attack_csv = config.METRICS_DIR / "la_attack_breakdown.csv"
    attack_df.to_csv(attack_csv, index=False)
    print("\n--- Granular Unseen Attack Breakdown (A07 - A19) ---")
    print(attack_df.to_string(index=False))

    # -------------------------------------------------------------
    # 6. Controlled Acoustic Stress Conditions
    # -------------------------------------------------------------
    conditions = [
        ("Clean", "clean"),
        ("Noise 20 dB", "noise_20db"),
        ("Noise 10 dB", "noise_10db"),
        ("Noise 5 dB", "noise_5db"),
        ("Reverberation", "reverberation"),
        ("Noise + Reverberation", "noise_reverb"),
        ("Low-pass Filter (3.4 kHz)", "lowpass_filter")
    ]

    bona_sample = df_eval[df_eval["label"] == 0].sample(n=min(n_robustness_sample, len(df_eval[df_eval["label"] == 0])),
                                                        random_state=config.RANDOM_SEED)
    spoof_sample = df_eval[df_eval["label"] == 1].sample(n=min(n_robustness_sample, len(df_eval[df_eval["label"] == 1])),
                                                         random_state=config.RANDOM_SEED)
    paired_stress_df = pd.concat([bona_sample, spoof_sample]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    print(f"\nRunning Paired Acoustic Robustness Stress Testing ({len(paired_stress_df)} utterances)...")
    robustness_rows = []

    for cond_name, cond_key in conditions:
        row = {"Condition": cond_name}
        for model_name, model in loaded_models.items():
            all_c_scores, all_c_labels = [], []
            with torch.no_grad():
                for _, r in paired_stress_df.iterrows():
                    audio = load_and_standardize_audio(r["filepath"])
                    corrupted = augmentation.corrupt_audio(audio, condition=cond_key)
                    w_t = torch.from_numpy(corrupted).unsqueeze(0)
                    lfcc = _LFCC_EXTRACTOR(w_t).to(device)
                    mel = _MEL_EXTRACTOR(w_t).to(device)
                    logit = model(lfcc, mel)

                    if logit.dim() > 0 and logit.shape[-1] == 2:
                        s = torch.softmax(logit, dim=-1)[..., 1].item()
                    else:
                        s = torch.sigmoid(logit.squeeze(-1)).item()

                    all_c_scores.append(s)
                    all_c_labels.append(int(r["label"]))

            all_c_scores = np.array(all_c_scores)
            all_c_labels = np.array(all_c_labels)

            fpr, tpr, _ = roc_curve(all_c_labels, all_c_scores)
            c_auc = float(auc(fpr, tpr))
            c_eer, _ = evaluate.compute_eer(all_c_labels, all_c_scores)
            preds_c = (all_c_scores >= 0.5).astype(int)
            cm_c = evaluate.compute_confusion_matrix_breakdown(all_c_labels, preds_c)

            row[f"{model_name} AUC"] = c_auc
            row[f"{model_name} EER (%)"] = c_eer * 100.0
            row[f"{model_name} F1"] = cm_c["F1"]

        if "Baseline CNN" in loaded_models and "Augmented CNN" in loaded_models:
            row["AUC Lift"] = row["Augmented CNN AUC"] - row["Baseline CNN AUC"]
            row["EER Reduction (%)"] = row["Baseline CNN EER (%)"] - row["Augmented CNN EER (%)"]

        robustness_rows.append(row)

    robustness_df = pd.DataFrame(robustness_rows)
    robustness_csv = config.METRICS_DIR / "la_robustness_comparison.csv"
    robustness_df.to_csv(robustness_csv, index=False)
    print("\n--- Controlled Acoustic Stress Test Results ---")
    print(robustness_df.to_string(index=False))

    # -------------------------------------------------------------
    # 7. Visualization Generation (100% Data-Driven)
    # -------------------------------------------------------------
    print("\nGenerating synchronized research figures...")

    # Figure 04: Confusion Matrix on Official LA Evaluation Set (Baseline)
    base_data = model_preds["Baseline CNN"]
    preds_base = (base_data["scores"] >= 0.5).astype(int)
    cm_info = evaluate.compute_confusion_matrix_breakdown(base_data["labels"], preds_base)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    cm_norm = cm_info["CM_row_norm_pct"]
    cm_raw = cm_info["CM_raw"]
    sns.heatmap(cm_norm, annot=False, cmap='Blues', vmin=0, vmax=100, ax=ax,
                xticklabels=['Bona Fide (0)', 'Spoof (1)'],
                yticklabels=['Bona Fide (0)', 'Spoof (1)'])

    cm_text = [
        [f"TN: {cm_raw[0, 0]}\n({cm_norm[0, 0]:.1f}%)", f"FP: {cm_raw[0, 1]}\n({cm_norm[0, 1]:.1f}%)"],
        [f"FN: {cm_raw[1, 0]}\n({cm_norm[1, 0]:.1f}%)", f"TP: {cm_raw[1, 1]}\n({cm_norm[1, 1]:.1f}%)"]
    ]
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_norm[i, j] > 50 else "black"
            ax.text(j + 0.5, i + 0.5, cm_text[i][j], ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    ax.set_title("Confusion Matrix: Official ASVspoof 2019 LA Evaluation\n(Dual-Stream MFM-LCNN Baseline — Row-Normalized %)",
                 fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "04_confusion_matrix_evaluation.png", dpi=300)
    plt.close()

    # Figure 05: Official ROC Curves
    plt.figure(figsize=(7.5, 6.5), dpi=300)
    colors = {"Baseline CNN": "#2980B9", "Augmented CNN": "#C0392B"}
    for name, data in model_preds.items():
        fpr, tpr, _ = roc_curve(data["labels"], data["scores"])
        roc_auc = auc(fpr, tpr)
        eer_val, _ = evaluate.compute_eer(data["labels"], data["scores"])
        c = colors.get(name, "#27AE60")
        plt.plot(fpr, tpr, lw=2.5, color=c, label=f"{name} (AUC = {roc_auc:.4f}, EER = {eer_val*100:.2f}%)")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (Bona Fide False Alarm)", fontsize=11, fontweight='bold')
    plt.ylabel("True Positive Rate (Spoof Detection Rate)", fontsize=11, fontweight='bold')
    plt.title("Official ASVspoof 2019 LA Evaluation ROC Curves\n(Unseen Attack Systems A07–A19)", fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "05_roc_curves_official_la.png", dpi=300)
    plt.close()

    # Figure 06: Per-Attack EER Bar Chart
    plt.figure(figsize=(12, 5.5), dpi=300)
    x = np.arange(len(eval_attacks))
    width = 0.35
    plt.bar(x - width/2, attack_df["Baseline CNN EER (%)"], width, label="Clean Baseline Dual LCNN", color="#2980B9")
    plt.bar(x + width/2, attack_df["Augmented CNN EER (%)"], width, label="Acoustically Hardened Dual LCNN", color="#C0392B")
    plt.xlabel("ASVspoof 2019 Unseen Attack ID (A07 - A19)", fontsize=11, fontweight='bold')
    plt.ylabel("Equal Error Rate (EER %)", fontsize=11, fontweight='bold')
    plt.title("Granular Breakdown: Unseen Attacks (A07–A19) on ASVspoof 2019 LA", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, [f"{a}\n({attack_df.loc[i, 'Type']})" for i, a in enumerate(eval_attacks)], rotation=0, fontsize=9, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "06_unseen_attacks_eer_barchart.png", dpi=300)
    plt.close()

    # Figure 07: Per-Attack AUC Bar Chart
    plt.figure(figsize=(12, 5.5), dpi=300)
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
    plt.savefig(config.FIGURES_DIR / "07_unseen_attacks_auc_barchart.png", dpi=300)
    plt.close()

    # Figure 08: Acoustic Robustness Degradation
    cond_labels = [c[0] for c in conditions]
    plt.figure(figsize=(11, 5.5), dpi=300)
    plt.plot(cond_labels, robustness_df["Baseline CNN AUC"], marker='o', lw=2.5, color='#2980B9', label="Clean Baseline Dual LCNN")
    plt.plot(cond_labels, robustness_df["Augmented CNN AUC"], marker='s', lw=2.5, color='#C0392B', label="Acoustically Hardened Dual LCNN")
    plt.title("Acoustic Robustness: Controlled Stress Conditions (ASVspoof 2019 LA)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Acoustic Stress Condition", fontsize=11, fontweight='bold')
    plt.ylabel("ROC-AUC Score", fontsize=11, fontweight='bold')
    plt.ylim(0.40, 1.02)
    plt.xticks(rotation=15, ha='right', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "08_acoustic_robustness_stress_testing.png", dpi=300)
    plt.close()

    print("\nAll figures and metric tables successfully regenerated directly from raw model inferences!")

if __name__ == "__main__":
    run_evaluation_pipeline()
