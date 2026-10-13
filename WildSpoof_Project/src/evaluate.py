import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def compute_eer(y_true, y_scores):
    """
    Calculate Equal Error Rate (EER) and decision threshold where False Positive Rate == False Negative Rate.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # EER is point where FPR == FNR
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    threshold = interp1d(fpr, thresholds)(eer)
    return eer, threshold

def evaluate_predictions(y_true, y_pred, y_scores, model_name="Model", feature_name="Feature", save_figures=True):
    """
    Comprehensive evaluation function returning metric dictionary and saving plots.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.5

    try:
        eer, eer_threshold = compute_eer(y_true, y_scores)
    except Exception as e:
        print(f"Warning: EER calculation failed ({e}), setting EER to 0.5")
        eer, eer_threshold = 0.5, 0.5

    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "Model": model_name,
        "Features": feature_name,
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "ROC-AUC": float(auc),
        "EER": float(eer),
        "EER Threshold": float(eer_threshold)
    }

    print(f"\n================ Evaluation Results: {model_name} ({feature_name}) ================")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"EER       : {eer * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("========================================================================\n")

    if save_figures:
        save_plots(y_true, y_scores, cm, model_name, feature_name, metrics)

    return metrics

def save_plots(y_true, y_scores, cm, model_name, feature_name, metrics):
    sns.set_theme(style="white")

    filename_prefix = f"{model_name.lower().replace(' ', '_')}_{feature_name.lower().replace(' ', '_')}"

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Bonafide (0)', 'Spoof (1)'],
                yticklabels=['Bonafide (0)', 'Spoof (1)'])
    plt.title(f"Confusion Matrix — {model_name} ({feature_name})", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"{filename_prefix}_confusion_matrix.png", dpi=300)
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {metrics['ROC-AUC']:.4f}, EER = {metrics['EER']*100:.2f}%)")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.plot([metrics['EER']], [1 - metrics['EER']], marker='o', markersize=8, color='red', label=f"EER Point ({metrics['EER']*100:.2f}%)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11)
    plt.title(f"ROC Curve — {model_name} ({feature_name})", fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"{filename_prefix}_roc_curve.png", dpi=300)
    plt.close()
