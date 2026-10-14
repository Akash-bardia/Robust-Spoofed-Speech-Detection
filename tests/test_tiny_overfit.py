import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "WildSpoof_Project"))
from src import config, features
from src.la_models import DualStreamLCNN

def test_tiny_overfit_pipeline():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Generate 20 synthetic audio signals (10 bona fide, 10 spoof)
    sr = 16000
    n_samples = 32000
    t = np.linspace(0, 2.0, n_samples, endpoint=False, dtype=np.float32)

    audio_list = []
    labels_list = []

    # Bona Fide (Label 0): Pure harmonic speech-like tones
    for i in range(10):
        f0 = 120.0 + i * 10.0
        wave = (0.5 * np.sin(2 * np.pi * f0 * t) +
                0.25 * np.sin(2 * np.pi * 2 * f0 * t) +
                0.12 * np.sin(2 * np.pi * 3 * f0 * t)).astype(np.float32)
        audio_list.append(wave)
        labels_list.append(0)

    # Spoof (Label 1): Modulated signals with high-frequency vocoder buzz
    for i in range(10):
        f0 = 150.0 + i * 10.0
        buzz = (0.4 * np.sin(2 * np.pi * f0 * t) +
                0.3 * np.sin(2 * np.pi * 4500 * t) +
                0.2 * np.random.normal(0, 0.1, n_samples)).astype(np.float32)
        audio_list.append(buzz)
        labels_list.append(1)

    audio_batch = torch.from_numpy(np.stack(audio_list))  # (20, 32000)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)  # (20,)

    # 2. Extract features through canonical pipeline
    lfcc_extractor = features.LFCCExtractor(n_filter=60, n_lfcc=60)
    mel_extractor = features.LogMelExtractor(n_mels=80)

    with torch.no_grad():
        lfcc_tensors = lfcc_extractor(audio_batch)  # (20, 1, 60, 201)
        mel_tensors = mel_extractor(audio_batch)    # (20, 1, 80, 201)

    assert lfcc_tensors.shape == (20, 1, 60, 201)
    assert mel_tensors.shape == (20, 1, 80, 201)

    # 3. Instantiate Dual-Stream LCNN Model (64-D per branch -> 128-D fused -> 2 classes)
    model = DualStreamLCNN(embedding_dim=64, num_classes=2, dropout=0.0)  # dropout 0 for deterministic overfit
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    # 4. Train for 25 epochs
    model.train()
    initial_loss = None
    final_loss = None

    for epoch in range(1, 26):
        optimizer.zero_grad()
        logits = model(lfcc_tensors, mel_tensors)
        loss = criterion(logits, labels_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch == 1:
            initial_loss = loss.item()
        final_loss = loss.item()

    # 5. Evaluate predictions
    model.eval()
    with torch.no_grad():
        test_logits = model(lfcc_tensors, mel_tensors)
        test_preds = torch.argmax(test_logits, dim=1)
        accuracy = (test_preds == labels_tensor).float().mean().item()

    print(f"\nTiny Overfit Test Results:")
    print(f"  Initial Loss: {initial_loss:.4f}")
    print(f"  Final Loss  : {final_loss:.4f}")
    print(f"  Accuracy    : {accuracy * 100.0:.2f}%")

    assert initial_loss > 0.40, f"Expected initial cross entropy ~0.69, got {initial_loss}"
    assert final_loss < 0.15, f"Model failed to overfit tiny dataset, final loss: {final_loss}"
    assert accuracy >= 0.95, f"Model did not achieve >95% accuracy on tiny dataset: {accuracy}"

if __name__ == "__main__":
    test_tiny_overfit_pipeline()
    print("test_tiny_overfit passed successfully!")
