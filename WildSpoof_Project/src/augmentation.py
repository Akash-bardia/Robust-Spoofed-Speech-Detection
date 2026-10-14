import numpy as np
import scipy.signal as signal
import torch
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def add_gaussian_noise(audio, snr_db=10.0):
    """
    Add additive Gaussian white noise to audio at an exact target Signal-to-Noise Ratio (SNR in dB).
    Mathematical Formulation:
      P_signal = (1 / N) * sum(x[t]^2)
      SNR_linear = 10^(SNR_dB / 10) = P_signal / P_noise
      P_noise = P_signal / SNR_linear
      noise ~ N(0, sqrt(P_noise))
      y[t] = x[t] + noise[t]
    Note: Labeled strictly as Additive Gaussian White Noise (not ambient/environmental noise).
    """
    audio = np.asarray(audio, dtype=np.float32)
    signal_power = np.mean(audio ** 2)
    if signal_power <= 1e-12:
        return audio

    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=len(audio)).astype(np.float32)
    noisy_audio = audio + noise
    return noisy_audio

def apply_reverberation(audio, sr=config.SAMPLE_RATE, rir_duration=0.2, decay_rate=15.0, rir_file=None):
    """
    Simulate room reverberation by convolving clean audio with a room impulse response (RIR):
      y(t) = x(t) * h(t)
    If rir_file is provided, loads and convolves with the recorded RIR.
    Otherwise, synthesizes a controlled stochastic exponential-decay impulse response:
      h(t) = N(0, 1) * exp(-decay_rate * t), for t in [0, rir_duration]
    The convolved output is trimmed to the original length and normalized to prevent clipping.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if rir_file is not None and Path(rir_file).exists():
        import soundfile as sf
        rir, rir_sr = sf.read(rir_file, dtype='float32')
        if rir.ndim > 1:
            rir = np.mean(rir, axis=1)
        if rir_sr != sr:
            import librosa
            rir = librosa.resample(rir, orig_sr=rir_sr, target_sr=sr)
    else:
        rir_samples = max(int(sr * rir_duration), 16)
        t = np.linspace(0.0, rir_duration, rir_samples, dtype=np.float32)
        rir = np.random.normal(0.0, 1.0, rir_samples).astype(np.float32) * np.exp(-decay_rate * t)

    max_rir = np.max(np.abs(rir))
    if max_rir > 1e-8:
        rir = rir / max_rir

    reverberated = signal.fftconvolve(audio, rir, mode='full')[:len(audio)].astype(np.float32)
    # Peak-normalize to prevent clipping
    max_rev = np.max(np.abs(reverberated))
    if max_rev > 1.0:
        reverberated = reverberated / max_rev
    return reverberated

def apply_lowpass_filter(audio, sr=config.SAMPLE_RATE, cutoff_freq=3400.0, order=4):
    """
    Apply a Butterworth low-pass filter (default cutoff 3.4 kHz to emulate narrowband telephony band limitation).
    Implemented via Second-Order Sections (SOS) for maximal numerical stability.
    """
    audio = np.asarray(audio, dtype=np.float32)
    nyquist = 0.5 * sr
    normal_cutoff = min(float(cutoff_freq) / nyquist, 0.999)
    sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    filtered = signal.sosfilt(sos, audio).astype(np.float32)
    return filtered

def apply_specaugment(feat_tensor, max_f=12, max_t=25, p_mask=0.45):
    """
    Apply 2D SpecAugment (frequency and time masking) directly to spectrogram representations.
    Prevents the network from memorizing fixed spectral artifact bands from known training attacks.
    Applied strictly to spectrogram tensors during training, NEVER to raw waveforms.
    Input shape: (..., freq_bins, time_frames)
    """
    if not isinstance(feat_tensor, torch.Tensor):
        feat = torch.from_numpy(feat_tensor).clone()
    else:
        feat = feat_tensor.clone()

    f_bins = feat.shape[-2]
    t_frames = feat.shape[-1]

    # Frequency Masking
    if np.random.rand() < p_mask:
        f_max_allowed = min(max_f, max(f_bins // 4, 1))
        f_mask = np.random.randint(2, f_max_allowed + 1)
        f0 = np.random.randint(0, f_bins - f_mask)
        feat[..., f0:f0+f_mask, :] = 0.0

    # Time Masking
    if np.random.rand() < p_mask:
        t_max_allowed = min(max_t, max(t_frames // 4, 1))
        t_mask = np.random.randint(4, t_max_allowed + 1)
        t0 = np.random.randint(0, t_frames - t_mask)
        feat[..., :, t0:t0+t_mask] = 0.0

    return feat

def corrupt_audio(audio, condition="clean", sr=config.SAMPLE_RATE):
    """
    Apply controlled acoustic stress corruption to an input waveform:
    - 'clean': Uncorrupted pristine speech
    - 'noise_20db': Additive Gaussian white noise at 20 dB SNR
    - 'noise_10db': Additive Gaussian white noise at 10 dB SNR
    - 'noise_5db': Additive Gaussian white noise at 5 dB SNR
    - 'reverberation': Synthetic RIR convolution (decay_rate = 15.0)
    - 'noise_reverb': Combined 10 dB SNR Gaussian noise + Reverberation
    - 'lowpass_filter' (or 'lowpass'): 3.4 kHz Butterworth 4th-order low-pass filter
    """
    cond = condition.lower()
    if cond == "clean":
        return audio
    elif cond in ["noise_20db", "noise_20"]:
        return add_gaussian_noise(audio, snr_db=20.0)
    elif cond in ["noise_10db", "noise_10"]:
        return add_gaussian_noise(audio, snr_db=10.0)
    elif cond in ["noise_5db", "noise_5"]:
        return add_gaussian_noise(audio, snr_db=5.0)
    elif cond in ["reverberation", "reverb"]:
        return apply_reverberation(audio, sr=sr)
    elif cond in ["noise_reverb", "noise_reverberation"]:
        noisy = add_gaussian_noise(audio, snr_db=10.0)
        return apply_reverberation(noisy, sr=sr)
    elif cond in ["lowpass_filter", "lowpass", "lowpass_4k"]:
        return apply_lowpass_filter(audio, sr=sr, cutoff_freq=3400.0)
    else:
        raise ValueError(f"Unknown corruption condition: {condition}")
