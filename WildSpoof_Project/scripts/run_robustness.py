import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, preprocessing, features, models, train, evaluate, augmentation, utils

def run_robustness_experiments(max_test_samples=500):
    utils.set_seed(config.RANDOM_SEED)

    print("Running Robustness / In-the-Wild Simulation Experiments...")

    # Load test manifest
    manifest_path = config.MANIFESTS_DIR / "test.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Test manifest missing: {manifest_path}")

    df_test = pd.read_csv(manifest_path)
    if max_test_samples is not None and len(df_test) > max_test_samples:
        df_test = df_test.sample(n=max_test_samples, random_state=config.RANDOM_SEED)

    # Load trained PyTorch CNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.SpoofCNN().to(device)
    checkpoint_path = config.MODELS_DIR / "best_cnn_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint missing: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    conditions = [
        ("Clean", "clean"),
        ("Noise 20 dB", "noise_20db"),
        ("Noise 10 dB", "noise_10db"),
        ("Noise 5 dB", "noise_5db"),
        ("Reverberation", "reverberation"),
        ("Noise + Reverberation", "noise_reverb")
    ]

    robustness_metrics = []

    for label_display, cond_key in conditions:
        print(f"\n--- Testing Robustness Condition: {label_display} ---")
        mel_list, label_list = [], []

        for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc=f"Evaluating {label_display}"):
            filepath = row["filepath"]
            label = row["label"]

            # Load and preprocess
            audio = preprocessing.preprocess_audio(filepath)
            # Apply acoustic corruption
            corrupted = augmentation.corrupt_audio(audio, condition=cond_key)
            # Extract log-Mel spectrogram
            mel = features.extract_mel_spectrogram(corrupted)

            mel_list.append(mel)
            label_list.append(label)

        X_mel_cond = np.array(mel_list)
        y_test_cond = np.array(label_list)

        # Predict
        preds, scores = train.predict_cnn(model, device, X_mel_cond)

        metrics = evaluate.evaluate_predictions(
            y_test_cond, preds, scores,
            model_name="CNN", feature_name=label_display,
            save_figures=True
        )

        robustness_metrics.append({
            "Condition": label_display,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "ROC-AUC": metrics["ROC-AUC"],
            "EER": metrics["EER"]
        })

    robustness_df = pd.DataFrame(robustness_metrics)
    robustness_csv_path = config.METRICS_DIR / "robustness_results.csv"
    robustness_df.to_csv(robustness_csv_path, index=False)

    print("\n================ Robustness Testing Final Table ================")
    print(robustness_df.to_string(index=False))
    print("================================================================\n")
    print(f"Robustness results saved to {robustness_csv_path}")

    return robustness_df

if __name__ == "__main__":
    run_robustness_experiments()
