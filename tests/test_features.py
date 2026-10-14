import pytest
import numpy as np
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "WildSpoof_Project"))
from src import config, features

def test_lfcc_mathematical_properties():
    sr = 16000
    n_fft = 512
    n_filter = 60
    n_lfcc = 60

    extractor = features.LFCCExtractor(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=400,
        hop_length=160,
        n_filter=n_filter,
        n_lfcc=n_lfcc,
        f_min=0.0,
        f_max=8000.0
    )

    # 1. Output shape assertion
    x = torch.randn(2, 32000)
    lfcc_out = extractor(x)
    assert lfcc_out.shape == (2, 1, n_lfcc, 201), f"Expected (2, 1, {n_lfcc}, 201), got {lfcc_out.shape}"
    assert torch.all(torch.isfinite(lfcc_out)), "LFCC output contains NaN or Inf"

    # 2. Filterbank covers intended frequency range (0 to 8000 Hz)
    fbank_mat = extractor.lfcc_transform.filter_mat  # Shape: (257, 60)
    assert fbank_mat.shape == (n_fft // 2 + 1, n_filter), f"Filter matrix shape mismatch: {fbank_mat.shape}"
    assert extractor.f_min == 0.0
    assert extractor.f_max == 8000.0

    # 3. Filterbank is linearly spaced across Hz spectrum
    # Find peak FFT bin for each filter
    peak_bins = fbank_mat.argmax(dim=0).cpu().numpy()
    bin_hz = peak_bins * (sr / float(n_fft))
    hz_diffs = np.diff(bin_hz)
    # Differences between adjacent filter centers should be nearly constant (linear spacing ~131.15 Hz)
    # Because discrete FFT bin resolution is 16000 / 512 = 31.25 Hz, adjacent peak bins step by either 4 or 5 bins (125 or 156.25 Hz)
    assert np.allclose(hz_diffs, 131.15, atol=32.0), f"Filterbank is not linearly spaced: {hz_diffs}"
    assert np.std(hz_diffs) < 16.0, f"Filterbank variance too high for linear spacing: {np.std(hz_diffs)}"

    # 4. Verify high-frequency bins (>4 kHz) are NOT truncated and actually contribute
    # Higher half of FFT bins (indices 128 to 256) must have non-zero filterbank weights
    high_freq_weights = fbank_mat[128:, :].sum().item()
    assert high_freq_weights > 0.0, "High frequency bins (>4 kHz) are missing from LFCC filterbank!"

    # 5. DCT matrix is actually applied
    dct_mat = extractor.lfcc_transform.dct_mat
    assert dct_mat.shape == (n_filter, n_lfcc), f"DCT matrix shape mismatch: {dct_mat.shape}"
    # Verify orthogonality of DCT matrix: DCT @ DCT.T ≈ Identity
    ident = torch.matmul(dct_mat, dct_mat.t())
    eye = torch.eye(n_filter, dtype=ident.dtype)
    assert torch.allclose(ident, eye, atol=1e-4), "DCT matrix is not orthonormal"

def test_logmel_properties():
    sr = 16000
    n_mels = 80
    extractor = features.LogMelExtractor(
        sample_rate=sr,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000.0
    )

    x = torch.randn(2, 32000)
    mel_out = extractor(x)
    assert mel_out.shape == (2, 1, n_mels, 201), f"Expected (2, 1, {n_mels}, 201), got {mel_out.shape}"
    assert torch.all(torch.isfinite(mel_out)), "Log-Mel output contains NaN or Inf"
    assert extractor.mel_transform.mel_scale.fb.shape == (257, n_mels)

if __name__ == "__main__":
    test_lfcc_mathematical_properties()
    test_logmel_properties()
    print("test_features passed successfully!")
