import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def compute_eer(y_true, y_scores):
    """
    Calculate Equal Error Rate (EER) and the decision threshold where False Alarm Rate == Miss Rate.
    Conventions:
      y_true: 0 = Bona Fide, 1 = Spoof
      y_scores: Continuous spoof score (higher means more likely spoof)
      False Alarm (P_fa / FPR): Bona fide score >= threshold
      Miss (P_miss / FNR): Spoof score < threshold
    Algorithm:
      Monotonic linear interpolation between discrete ROC operating thresholds.
      Guaranteed to converge, never throws numerical root-finding errors, and handles ties robustly.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5

    bona_scores = y_scores[y_true == 0]
    spoof_scores = y_scores[y_true == 1]

    n_bona = len(bona_scores)
    n_spoof = len(spoof_scores)

    if n_bona == 0 or n_spoof == 0:
        return 0.5, 0.5

    # Discrete candidate thresholds from sorted scores
    thresholds = np.sort(np.unique(y_scores))
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])

    bona_sorted = np.sort(bona_scores)
    spoof_sorted = np.sort(spoof_scores)

    # P_fa(t): fraction of bona fide scores >= t
    p_fa = (n_bona - np.searchsorted(bona_sorted, thresholds, side='left')) / float(n_bona)
    # P_miss(t): fraction of spoof scores < t
    p_miss = np.searchsorted(spoof_sorted, thresholds, side='left') / float(n_spoof)

    diff = p_fa - p_miss
    # Find crossing where diff transitions from positive to non-positive
    idx = np.where(diff <= 0)[0]
    if len(idx) == 0:
        return 0.0, float(thresholds[-1])
    k = idx[0]
    if k == 0:
        return 0.0, float(thresholds[0])

    # Linear interpolation between thresholds k-1 and k
    denom = (p_fa[k - 1] - p_fa[k]) + (p_miss[k] - p_miss[k - 1])
    if denom == 0:
        eer = p_fa[k]
        thresh = thresholds[k]
    else:
        frac = (p_fa[k - 1] - p_miss[k - 1]) / denom
        eer = p_fa[k - 1] - frac * (p_fa[k - 1] - p_fa[k])
        if np.isfinite(thresholds[k - 1]) and np.isfinite(thresholds[k]):
            thresh = thresholds[k - 1] + frac * (thresholds[k] - thresholds[k - 1])
        else:
            thresh = thresholds[k]

    return float(np.clip(eer, 0.0, 1.0)), float(thresh)

def compute_confusion_matrix_breakdown(y_true, y_pred):
    """
    Compute rigorous binary confusion matrix counts, rates, and row-normalized percentages.
    Conventions:
      0 = Bona Fide (Negative class)
      1 = Spoof (Positive class)
    Cells:
      TN: True Bona Fide (label 0, pred 0)
      FP: False Alarm (label 0, pred 1)
      FN: Missed Spoof (label 1, pred 0)
      TP: Detected Spoof (label 1, pred 1)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = int(tn + fp + fn + tp)

    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Row-normalized percentages (per-class recall)
    bona_total = tn + fp
    spoof_total = fn + tp
    tn_pct = float(tn / bona_total * 100.0) if bona_total > 0 else 0.0
    fp_pct = float(fp / bona_total * 100.0) if bona_total > 0 else 0.0
    fn_pct = float(fn / spoof_total * 100.0) if spoof_total > 0 else 0.0
    tp_pct = float(tp / spoof_total * 100.0) if spoof_total > 0 else 0.0

    cm_norm = np.array([
        [tn_pct, fp_pct],
        [fn_pct, tp_pct]
    ])

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Total": total,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "CM_raw": cm,
        "CM_row_norm_pct": cm_norm
    }

def evaluate_predictions(y_true, y_pred, y_scores, model_name="Model", feature_name="Feature", save_figures=True):
    """
    Comprehensive evaluation returning verified metrics, confusion breakdown, and sanity checks.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)

    cm_info = compute_confusion_matrix_breakdown(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.5

    eer, eer_threshold = compute_eer(y_true, y_scores)

    metrics = {
        "Model": model_name,
        "Features": feature_name,
        "Samples": int(len(y_true)),
        "Accuracy": float(cm_info["Accuracy"]),
        "Precision": float(cm_info["Precision"]),
        "Recall": float(cm_info["Recall"]),
        "Specificity": float(cm_info["Specificity"]),
        "F1": float(cm_info["F1"]),
        "ROC-AUC": float(auc),
        "EER": float(eer),
        "EER (%)": float(eer * 100.0),
        "EER Threshold": float(eer_threshold),
        "TN": cm_info["TN"],
        "FP": cm_info["FP"],
        "FN": cm_info["FN"],
        "TP": cm_info["TP"]
    }

    # Rigorous Sanity Checks
    assert 0.0 <= metrics["ROC-AUC"] <= 1.0, f"ROC-AUC out of bounds: {metrics['ROC-AUC']}"
    assert 0.0 <= metrics["EER"] <= 1.0, f"EER out of bounds: {metrics['EER']}"
    assert 0.0 <= metrics["Precision"] <= 1.0, f"Precision out of bounds: {metrics['Precision']}"
    assert 0.0 <= metrics["Recall"] <= 1.0, f"Recall out of bounds: {metrics['Recall']}"
    assert 0.0 <= metrics["Accuracy"] <= 1.0, f"Accuracy out of bounds: {metrics['Accuracy']}"
    assert (metrics["TN"] + metrics["FP"] + metrics["FN"] + metrics["TP"]) == len(y_true), "Confusion count mismatch"
    assert np.isfinite(metrics["ROC-AUC"]) and np.isfinite(metrics["EER"]), "NaN/Inf detected in metrics"

    print(f"\n================ Evaluation Results: {model_name} ({feature_name}) ================")
    print(f"Total Evaluated Samples: {metrics['Samples']}")
    print(f"Accuracy               : {metrics['Accuracy'] * 100.0:.2f}%")
    print(f"Precision (Spoof)      : {metrics['Precision']:.4f}")
    print(f"Recall (Spoof)         : {metrics['Recall']:.4f}")
    print(f"Specificity (Bona Fide): {metrics['Specificity']:.4f}")
    print(f"F1-Score               : {metrics['F1']:.4f}")
    print(f"ROC-AUC Score          : {metrics['ROC-AUC']:.4f}")
    print(f"Equal Error Rate (EER) : {metrics['EER (%)']:.2f}% (Threshold: {metrics['EER Threshold']:.4f})")
    print("Confusion Matrix Counts (TN, FP, FN, TP):")
    print(f"  Bona Fide [TN: {metrics['TN']:5d}, FP: {metrics['FP']:5d}]")
    print(f"  Spoof     [FN: {metrics['FN']:5d}, TP: {metrics['TP']:5d}]")
    print("Row-Normalized Percentages:")
    print(f"  Bona Fide Recall: {cm_info['CM_row_norm_pct'][0, 0]:.2f}%, False Alarm Rate: {cm_info['CM_row_norm_pct'][0, 1]:.2f}%")
    print(f"  Spoof Miss Rate : {cm_info['CM_row_norm_pct'][1, 0]:.2f}%, Spoof Recall     : {cm_info['CM_row_norm_pct'][1, 1]:.2f}%")
    print("========================================================================\n")

    if save_figures:
        save_plots(y_true, y_scores, cm_info, model_name, feature_name, metrics)

    return metrics

def save_plots(y_true, y_scores, cm_info, model_name, feature_name, metrics):
    """
    Generate and save verified confusion matrix heatmap and ROC curve.
    Both figures are produced directly from raw predictions without hardcoded figures.
    """
    sns.set_theme(style="white")
    filename_prefix = f"{model_name.lower().replace(' ', '_')}_{feature_name.lower().replace(' ', '_')}"

    # 1. Confusion Matrix Heatmap
    cm_raw = cm_info["CM_raw"]
    cm_norm = cm_info["CM_row_norm_pct"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(cm_norm, annot=False, cmap='Blues', vmin=0, vmax=100, ax=ax,
                xticklabels=['Bona Fide (0)', 'Spoof (1)'],
                yticklabels=['Bona Fide (0)', 'Spoof (1)'])

    # Annotate each cell with count and row-wise percentage
    labels = [
        [f"TN: {cm_raw[0, 0]}\n({cm_norm[0, 0]:.1f}%)", f"FP: {cm_raw[0, 1]}\n({cm_norm[0, 1]:.1f}%)"],
        [f"FN: {cm_raw[1, 0]}\n({cm_norm[1, 0]:.1f}%)", f"TP: {cm_raw[1, 1]}\n({cm_norm[1, 1]:.1f}%)"]
    ]
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_norm[i, j] > 50 else "black"
            ax.text(j + 0.5, i + 0.5, labels[i][j], ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    ax.set_title(f"Confusion Matrix: {model_name}\n({feature_name} — Row-Normalized %)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig_path = config.FIGURES_DIR / f"{filename_prefix}_confusion_matrix.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(fpr, tpr, color='#E67E22', lw=2.5,
             label=f"ROC curve (AUC = {metrics['ROC-AUC']:.4f}, EER = {metrics['EER (%)']:.2f}%)")
    plt.plot([0, 1], [0, 1], color='#34495E', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')
    plt.plot([metrics['EER']], [1.0 - metrics['EER']], marker='o', markersize=8, color='#C0392B',
             label=f"EER Operating Point ({metrics['EER (%)']:.2f}%)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
    plt.title(f"ROC Curve: {model_name} ({feature_name})", fontsize=12, fontweight='bold', pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    roc_path = config.FIGURES_DIR / f"{filename_prefix}_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
