import numpy as np
import scipy.signal as signal
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def add_gaussian_noise(audio, snr_db=10):
    """
    Add additive Gaussian white noise to audio at a target Signal-to-Noise Ratio (SNR in dB).
    """
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise

def apply_reverberation(audio, sr=config.SAMPLE_RATE, rir_duration=0.2, decay_rate=15.0):
    """
    Simulate room reverberation by convolving audio with a synthetic exponential decay RIR.
    """
    rir_samples = int(sr * rir_duration)
    t = np.linspace(0, rir_duration, rir_samples)
    rir = np.random.normal(0, 1, rir_samples) * np.exp(-decay_rate * t)
    rir = rir / np.max(np.abs(rir))

    reverberated = signal.fftconvolve(audio, rir, mode='full')[:len(audio)]
    return reverberated

def apply_amplitude_scaling(audio, scale_factor=0.5):
    """
    Scale audio amplitude by a constant factor.
    """
    return audio * scale_factor

def apply_lowpass_filter(audio, sr=config.SAMPLE_RATE, cutoff_freq=3400):
    """
    Apply a 4th-order Butterworth low-pass filter (e.g. 3.4 kHz cutoff to simulate narrowband telephony).
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered = signal.lfilter(b, a, audio)
    return filtered

def corrupt_audio(audio, condition="clean", sr=config.SAMPLE_RATE):
    """
    Apply specified acoustic degradation condition to input audio:
    - 'clean': No degradation
    - 'noise_20db': Gaussian Noise at 20 dB SNR
    - 'noise_10db': Gaussian Noise at 10 dB SNR
    - 'noise_5db': Gaussian Noise at 5 dB SNR
    - 'reverberation': Synthetic Room Reverberation
    - 'noise_reverb': 10 dB Noise + Reverberation
    - 'lowpass_filter': 3.4 kHz Low-pass Filter
    """
    if condition == "clean":
        return audio
    elif condition == "noise_20db":
        return add_gaussian_noise(audio, snr_db=20)
    elif condition == "noise_10db":
        return add_gaussian_noise(audio, snr_db=10)
    elif condition == "noise_5db":
        return add_gaussian_noise(audio, snr_db=5)
    elif condition == "reverberation":
        return apply_reverberation(audio, sr=sr)
    elif condition == "noise_reverb":
        noisy = add_gaussian_noise(audio, snr_db=10)
        return apply_reverberation(noisy, sr=sr)
    elif condition == "lowpass_filter":
        return apply_lowpass_filter(audio, sr=sr)
    else:
        raise ValueError(f"Unknown corruption condition: {condition}")
