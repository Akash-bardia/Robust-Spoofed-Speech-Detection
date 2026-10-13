import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
import joblib

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, dataset, train, evaluate, utils

def run_all_experiments():
    utils.set_seed(config.RANDOM_SEED)

    print("Loading cached train, validation, and test features...")
    X_train_mfcc, X_train_lfcc, X_train_mel, y_train = dataset.load_cached_data("train")
    X_val_mfcc, X_val_lfcc, X_val_mel, y_val = dataset.load_cached_data("validation")
    X_test_mfcc, X_test_lfcc, X_test_mel, y_test = dataset.load_cached_data("test")

    metrics_results = []

    # -------------------------------------------------------------
    # Model 1: SVM + MFCC
    # -------------------------------------------------------------
    print("\n--- Training Model 1: SVM + MFCC ---")
    scaler_mfcc = StandardScaler()
    X_train_mfcc_scaled = scaler_mfcc.fit_transform(X_train_mfcc)
    X_test_mfcc_scaled = scaler_mfcc.transform(X_test_mfcc)

    svm_mfcc = SVC(kernel='rbf', probability=True, C=1.0, random_state=config.RANDOM_SEED)
    svm_mfcc.fit(X_train_mfcc_scaled, y_train)

    y_scores_svm_mfcc = svm_mfcc.predict_proba(X_test_mfcc_scaled)[:, 1]
    y_preds_svm_mfcc = (y_scores_svm_mfcc >= 0.5).astype(int)

    metrics_svm_mfcc = evaluate.evaluate_predictions(
        y_test, y_preds_svm_mfcc, y_scores_svm_mfcc,
        model_name="SVM", feature_name="MFCC"
    )
    metrics_results.append(metrics_svm_mfcc)
    joblib.dump(svm_mfcc, config.MODELS_DIR / "svm_mfcc.pkl")
    joblib.dump(scaler_mfcc, config.MODELS_DIR / "scaler_mfcc.pkl")

    # -------------------------------------------------------------
    # Model 2: SVM + LFCC
    # -------------------------------------------------------------
    print("\n--- Training Model 2: SVM + LFCC ---")
    scaler_lfcc = StandardScaler()
    X_train_lfcc_scaled = scaler_lfcc.fit_transform(X_train_lfcc)
    X_test_lfcc_scaled = scaler_lfcc.transform(X_test_lfcc)

    svm_lfcc = SVC(kernel='rbf', probability=True, C=1.0, random_state=config.RANDOM_SEED)
    svm_lfcc.fit(X_train_lfcc_scaled, y_train)

    y_scores_svm_lfcc = svm_lfcc.predict_proba(X_test_lfcc_scaled)[:, 1]
    y_preds_svm_lfcc = (y_scores_svm_lfcc >= 0.5).astype(int)

    metrics_svm_lfcc = evaluate.evaluate_predictions(
        y_test, y_preds_svm_lfcc, y_scores_svm_lfcc,
        model_name="SVM", feature_name="LFCC"
    )
    metrics_results.append(metrics_svm_lfcc)
    joblib.dump(svm_lfcc, config.MODELS_DIR / "svm_lfcc.pkl")
    joblib.dump(scaler_lfcc, config.MODELS_DIR / "scaler_lfcc.pkl")

    # -------------------------------------------------------------
    # Model 3: GMM + LFCC
    # -------------------------------------------------------------
    print("\n--- Training Model 3: GMM + LFCC Baseline ---")
    X_train_real_lfcc = X_train_lfcc_scaled[y_train == 0]
    X_train_fake_lfcc = X_train_lfcc_scaled[y_train == 1]

    gmm_real = GaussianMixture(n_components=8, covariance_type='diag', random_state=config.RANDOM_SEED)
    gmm_fake = GaussianMixture(n_components=8, covariance_type='diag', random_state=config.RANDOM_SEED)

    gmm_real.fit(X_train_real_lfcc)
    gmm_fake.fit(X_train_fake_lfcc)

    # Log-likelihood ratio
    ll_real = gmm_real.score_samples(X_test_lfcc_scaled)
    ll_fake = gmm_fake.score_samples(X_test_lfcc_scaled)
    llr_scores = ll_fake - ll_real  # Higher score => higher likelihood of spoof

    # Min-Max normalize LLR to [0, 1] probability range for evaluation
    min_score, max_score = np.min(llr_scores), np.max(llr_scores)
    prob_scores_gmm = (llr_scores - min_score) / (max_score - min_score + 1e-10)
    y_preds_gmm = (llr_scores > 0).astype(int)

    metrics_gmm_lfcc = evaluate.evaluate_predictions(
        y_test, y_preds_gmm, prob_scores_gmm,
        model_name="GMM", feature_name="LFCC"
    )
    metrics_results.append(metrics_gmm_lfcc)
    joblib.dump(gmm_real, config.MODELS_DIR / "gmm_real.pkl")
    joblib.dump(gmm_fake, config.MODELS_DIR / "gmm_fake.pkl")

    # -------------------------------------------------------------
    # Model 4: PyTorch CNN + Mel Spectrogram
    # -------------------------------------------------------------
    print("\n--- Training Model 4: PyTorch CNN + Mel Spectrogram ---")
    cnn_model, device = train.train_cnn(
        X_train_mel, y_train, X_val_mel, y_val,
        batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE, epochs=config.EPOCHS
    )

    y_preds_cnn, y_scores_cnn = train.predict_cnn(cnn_model, device, X_test_mel)

    metrics_cnn = evaluate.evaluate_predictions(
        y_test, y_preds_cnn, y_scores_cnn,
        model_name="CNN", feature_name="Mel"
    )
    metrics_results.append(metrics_cnn)

    # Save Comparison Table
    results_df = pd.DataFrame(metrics_results)
    comparison_path = config.METRICS_DIR / "baseline_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    print("\n================ Final Baseline Comparison Table ================")
    print(results_df.to_string(index=False))
    print("=================================================================\n")
    print(f"Results saved to {comparison_path}")

    return results_df

if __name__ == "__main__":
    run_all_experiments()
