import numpy as np
import librosa
from scipy.fftpack import dct
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def compute_linear_filterbank(sr=config.SAMPLE_RATE, n_fft=config.N_FFT, n_filter=40, fmin=0, fmax=None):
    """
    Construct a linear-frequency filterbank matrix of shape (n_filter, n_fft // 2 + 1).
    Linear spacing across [fmin, fmax] frequency range.
    """
    if fmax is None:
        fmax = sr / 2.0

    # Linear spacing in Hz
    freq_points = np.linspace(fmin, fmax, n_filter + 2)
    # FFT bin indices corresponding to Hz frequencies
    bin_points = np.floor((n_fft + 1) * freq_points / sr).astype(int)

    num_bins = n_fft // 2 + 1
    fbank = np.zeros((n_filter, num_bins))

    for m in range(1, n_filter + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)

    return fbank

def extract_mfcc(audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC):
    """
    Extract MFCC features with delta and delta-delta.
    Aggregates frame-level features using mean and standard deviation.
    Output vector length = n_mfcc * 3 * 2 (e.g., 20*3*2 = 120 features).
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])  # (3 * n_mfcc, T)

    mean_feat = np.mean(combined, axis=1)
    std_feat = np.std(combined, axis=1)

    feature_vec = np.hstack([mean_feat, std_feat])
    return feature_vec

def extract_lfcc(audio, sr=config.SAMPLE_RATE, n_lfcc=config.N_LFCC, n_filter=40):
    """
    Extract Linear Frequency Cepstral Coefficients (LFCC) using a linear filterbank.
    Calculates LFCC + delta + delta-delta, aggregated by mean & std.
    """
    # 1. Short-Time Fourier Transform (STFT) magnitude spectrum
    stft = librosa.stft(
        audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH
    )
    mag_spec = np.abs(stft)  # (n_fft // 2 + 1, T)

    # 2. Apply Linear Filterbank
    fbank = compute_linear_filterbank(sr=sr, n_fft=config.N_FFT, n_filter=n_filter, fmin=config.FMIN, fmax=config.FMAX)
    linear_energies = np.dot(fbank, mag_spec)  # (n_filter, T)

    # 3. Log compression (add small epsilon to avoid log(0))
    log_energies = np.log(linear_energies + 1e-10)

    # 4. Discrete Cosine Transform (DCT-II) along filter axis
    lfcc = dct(log_energies, type=2, axis=0, norm='ortho')[:n_lfcc, :]  # (n_lfcc, T)

    # 5. Delta & Delta-Delta
    delta = librosa.feature.delta(lfcc)
    delta2 = librosa.feature.delta(lfcc, order=2)

    combined = np.vstack([lfcc, delta, delta2])  # (3 * n_lfcc, T)

    mean_feat = np.mean(combined, axis=1)
    std_feat = np.std(combined, axis=1)

    feature_vec = np.hstack([mean_feat, std_feat])
    return feature_vec

def extract_spectral_features(audio, sr=config.SAMPLE_RATE):
    """
    Extract classical spectral descriptors:
    - Spectral Centroid
    - Spectral Bandwidth
    - Spectral Rolloff
    - Zero Crossing Rate
    - RMS Energy
    Aggregated by mean and std (10 features).
    """
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=config.HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH)

    stacked = np.vstack([cent, bw, rolloff, zcr, rms])  # (5, T)
    mean_feat = np.mean(stacked, axis=1)
    std_feat = np.std(stacked, axis=1)

    return np.hstack([mean_feat, std_feat])

def extract_mel_spectrogram(audio, sr=config.SAMPLE_RATE, n_mels=config.N_MELS):
    """
    Extract Log-Mel Spectrogram image representation for CNN deep learning model.
    Output shape: (n_mels, time_steps)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH,
        fmin=config.FMIN, fmax=config.FMAX
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec
