import pytest
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "WildSpoof_Project"))
from src import evaluate

def test_eer_monotonic_interpolation():
    # 1. Perfect separation (Bona fide: low scores, Spoof: high scores)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.25, 0.3, 0.7, 0.8, 0.85, 0.9])
    eer, thresh = evaluate.compute_eer(y_true, y_scores)
    assert np.isclose(eer, 0.0), f"Expected EER 0.0 for perfect separation, got {eer}"

    # 2. Symmetric partial overlap
    # 2 bona fide: [0.2, 0.6], 2 spoof: [0.4, 0.8]
    # At t=0.5: Bona >= 0.5 is 1 (P_fa = 0.5), Spoof < 0.5 is 1 (P_miss = 0.5)
    y_true_sym = np.array([0, 0, 1, 1])
    y_scores_sym = np.array([0.2, 0.6, 0.4, 0.8])
    eer_sym, thresh_sym = evaluate.compute_eer(y_true_sym, y_scores_sym)
    assert np.isclose(eer_sym, 0.5), f"Expected EER 0.5, got {eer_sym}"

    # 3. Corner case: identical scores (complete ambiguity)
    y_true_same = np.array([0, 0, 1, 1])
    y_scores_same = np.array([0.5, 0.5, 0.5, 0.5])
    eer_same, _ = evaluate.compute_eer(y_true_same, y_scores_same)
    assert 0.0 <= eer_same <= 1.0

def test_confusion_matrix_breakdown():
    # Construct exact test cases:
    # 10 samples: 4 Bona Fide (0), 6 Spoof (1)
    # Bona Fide: 3 predicted 0 (TN), 1 predicted 1 (FP)
    # Spoof: 2 predicted 0 (FN), 4 predicted 1 (TP)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1])

    cm_info = evaluate.compute_confusion_matrix_breakdown(y_true, y_pred)
    assert cm_info["TN"] == 3
    assert cm_info["FP"] == 1
    assert cm_info["FN"] == 2
    assert cm_info["TP"] == 4
    assert cm_info["Total"] == 10

    # Accuracy = (3 + 4) / 10 = 0.70
    assert np.isclose(cm_info["Accuracy"], 0.70)
    # Precision = TP / (TP + FP) = 4 / (4 + 1) = 4/5 = 0.80
    assert np.isclose(cm_info["Precision"], 0.80)
    # Recall = TP / (TP + FN) = 4 / (4 + 2) = 4/6 = 2/3 ≈ 0.6667
    assert np.isclose(cm_info["Recall"], 4.0 / 6.0)
    # Specificity = TN / (TN + FP) = 3 / 4 = 0.75
    assert np.isclose(cm_info["Specificity"], 0.75)
    # F1 = 2 * (0.80 * 4/6) / (0.80 + 4/6)
    expected_f1 = 2 * (0.8 * (4/6)) / (0.8 + 4/6)
    assert np.isclose(cm_info["F1"], expected_f1)

    # Row-normalized matrix
    # Row 0: TN % = 75%, FP % = 25%
    # Row 1: FN % = 33.33%, TP % = 66.67%
    norm_mat = cm_info["CM_row_norm_pct"]
    assert np.isclose(norm_mat[0, 0], 75.0)
    assert np.isclose(norm_mat[0, 1], 25.0)
    assert np.isclose(norm_mat[1, 0], 100.0 / 3.0)
    assert np.isclose(norm_mat[1, 1], 200.0 / 3.0)

if __name__ == "__main__":
    test_eer_monotonic_interpolation()
    test_confusion_matrix_breakdown()
    print("test_metrics passed successfully!")
