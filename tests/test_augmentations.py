import pytest
import numpy as np
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "WildSpoof_Project"))
from src import config, augmentation

def test_gaussian_noise_snr():
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave signal with known power
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sig_power = np.mean(signal ** 2)

    for target_snr in [5.0, 10.0, 20.0]:
        noisy = augmentation.add_gaussian_noise(signal, snr_db=target_snr)
        noise = noisy - signal
        measured_noise_power = np.mean(noise ** 2)
        measured_snr = 10.0 * np.log10(sig_power / measured_noise_power)
        # Measured SNR should be within 0.5 dB of target SNR
        assert np.isclose(measured_snr, target_snr, atol=0.5), (
            f"Expected SNR {target_snr} dB, got {measured_snr:.2f} dB"
        )

def test_reverberation_convolution():
    sr = 16000
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    reverberated = augmentation.apply_reverberation(audio, sr=sr, rir_duration=0.2, decay_rate=15.0)
    assert len(reverberated) == len(audio), "Reverberated audio must preserve original length"
    assert np.all(np.isfinite(reverberated)), "Reverberation generated NaN or Inf"
    assert np.max(np.abs(reverberated)) <= 1.0, "Reverberated audio exceeded clipping boundary"

def test_lowpass_filter_attenuation():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 500 Hz low frequency tone (passband)
    low_tone = np.sin(2 * np.pi * 500 * t).astype(np.float32)
    # 6000 Hz high frequency tone (stopband, above 3400 Hz cutoff)
    high_tone = np.sin(2 * np.pi * 6000 * t).astype(np.float32)

    filt_low = augmentation.apply_lowpass_filter(low_tone, sr=sr, cutoff_freq=3400.0)
    filt_high = augmentation.apply_lowpass_filter(high_tone, sr=sr, cutoff_freq=3400.0)

    # 500 Hz power should be preserved (> 90%)
    p_low_orig = np.mean(low_tone ** 2)
    p_low_filt = np.mean(filt_low ** 2)
    assert p_low_filt / p_low_orig > 0.90, "Passband tone was unduly attenuated"

    # 6000 Hz power should be attenuated by > 99% (< 0.01 remaining power)
    p_high_orig = np.mean(high_tone ** 2)
    p_high_filt = np.mean(filt_high ** 2)
    assert p_high_filt / p_high_orig < 0.01, "Stopband tone (>3.4 kHz) was not sufficiently attenuated"

def test_specaugment_masking():
    # Shape: (1, 60, 201)
    feat = torch.ones(1, 60, 201)
    # Force mask by setting p_mask = 1.0
    masked = augmentation.apply_specaugment(feat, max_f=12, max_t=25, p_mask=1.0)
    assert masked.shape == feat.shape
    # Ensure some zeros are introduced
    zero_count = (masked == 0.0).sum().item()
    assert zero_count > 0, "SpecAugment should have zeroed masked regions"

if __name__ == "__main__":
    test_gaussian_noise_snr()
    test_reverberation_convolution()
    test_lowpass_filter_attenuation()
    test_specaugment_masking()
    print("test_augmentations passed successfully!")
