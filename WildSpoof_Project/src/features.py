import numpy as np
import librosa
from scipy.fftpack import dct
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as AF

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def compute_linear_filterbank(sr=config.SAMPLE_RATE, n_fft=config.N_FFT, n_filter=60, fmin=0.0, fmax=None):
    """
    Construct a linear-frequency triangular filterbank matrix of shape (n_filter, n_fft // 2 + 1).
    Uniformly (linearly) spaced across [fmin, fmax] frequency range up to Nyquist (sr / 2).
    """
    if fmax is None:
        fmax = sr / 2.0

    # Linearly spaced center frequencies in Hz
    freq_points = np.linspace(fmin, fmax, n_filter + 2)
    # FFT bin indices corresponding to Hz frequencies: bin = freq * n_fft / sr
    bin_points = np.round(freq_points * n_fft / sr).astype(int)

    num_bins = n_fft // 2 + 1
    fbank = np.zeros((n_filter, num_bins), dtype=np.float32)

    for m in range(1, n_filter + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus and k < num_bins:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m and k < num_bins:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank

class LFCCExtractor(nn.Module):
    """
    Mathematically rigorous Linear Frequency Cepstral Coefficients (LFCC) Extractor.
    Pipeline:
      Waveform (16 kHz, 32000 samples)
      -> Framing & Hann Windowing (win_length=400, hop_length=160, n_fft=512)
      -> Short-Time Fourier Transform (STFT) Power Spectrum: (B, 257, 201)
      -> LINEAR Frequency Filterbank (60 triangular bands spanning 0 to 8000 Hz): (257, 60)
      -> Logarithmic Compression (ln(energy + 1e-6))
      -> Discrete Cosine Transform (orthonormal DCT-II): (60, 60)
      -> 2D LFCC Representation: (B, 1, 60, 201)
    """
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                 n_filter=60, n_lfcc=60, f_min=0.0, f_max=8000.0):
        super(LFCCExtractor, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate / 2.0

        # Vectorized Torchaudio LFCC transform with exact canonical parameters
        self.lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            f_min=self.f_min,
            f_max=self.f_max,
            log_lf=True,
            speckwargs={
                'n_fft': n_fft,
                'win_length': win_length,
                'hop_length': hop_length,
                'center': True,
                'power': 2.0
            }
        )

        # Assert correct filter and DCT matrix shapes
        assert self.lfcc_transform.filter_mat.shape == (n_fft // 2 + 1, n_filter), (
            f"Filter matrix shape mismatch: {self.lfcc_transform.filter_mat.shape}"
        )
        assert self.lfcc_transform.dct_mat.shape == (n_filter, n_lfcc), (
            f"DCT matrix shape mismatch: {self.lfcc_transform.dct_mat.shape}"
        )

    def forward(self, x):
        """
        Input: Tensor of shape (..., time) or (B, 1, time)
        Output: Tensor of shape (B, 1, n_lfcc, time_frames)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, time)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)   # (B, time)

        lfcc = self.lfcc_transform(x)  # (B, n_lfcc, time_frames)
        if lfcc.dim() == 2:
            lfcc = lfcc.unsqueeze(0)
        if lfcc.dim() == 3:
            lfcc = lfcc.unsqueeze(1)  # (B, 1, n_lfcc, time_frames)

        assert lfcc.shape[-2] == self.n_lfcc, f"Expected {self.n_lfcc} LFCC bands, got {lfcc.shape[-2]}"
        return lfcc

class LogMelExtractor(nn.Module):
    """
    Log-Mel Spectrogram Extractor for perceptual acoustic representations.
    Pipeline:
      Waveform (16 kHz, 32000 samples)
      -> Framing & Hann Windowing (win_length=400, hop_length=160, n_fft=512)
      -> STFT Power Spectrum: (B, 257, 201)
      -> Mel Filterbank (80 perceptual Mel bands spanning 0 to 8000 Hz): (257, 80)
      -> Logarithmic Compression (AmplitudeToDB)
      -> 2D Log-Mel Representation: (B, 1, 80, 201)
    """
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
                 n_mels=80, f_min=0.0, f_max=8000.0):
        super(LogMelExtractor, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0
        )
        self.amp_to_db = T.AmplitudeToDB()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        mel = self.amp_to_db(self.mel_transform(x))
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)

        assert mel.shape[-2] == self.n_mels, f"Expected {self.n_mels} Mel bands, got {mel.shape[-2]}"
        return mel

# Global singleton extractors for optimal throughput
_GLOBAL_LFCC_EXTRACTOR = LFCCExtractor(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    win_length=config.WIN_LENGTH,
    hop_length=config.HOP_LENGTH,
    n_filter=config.N_LFCC_FILTERS,
    n_lfcc=config.N_LFCC,
    f_min=config.FMIN,
    f_max=config.FMAX
)

_GLOBAL_MEL_EXTRACTOR = LogMelExtractor(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    win_length=config.WIN_LENGTH,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS,
    f_min=config.FMIN,
    f_max=config.FMAX
)

def extract_2d_lfcc(audio, sr=config.SAMPLE_RATE):
    """
    Extract 2D LFCC spectrogram tensor of shape (1, 60, 201) or (60, 201).
    """
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
    else:
        audio_tensor = audio
    with torch.no_grad():
        lfcc = _GLOBAL_LFCC_EXTRACTOR(audio_tensor).squeeze(0).squeeze(0).numpy()
    return lfcc.astype(np.float32)

def extract_2d_mel(audio, sr=config.SAMPLE_RATE):
    """
    Extract 2D Log-Mel spectrogram tensor of shape (1, 80, 201) or (80, 201).
    """
    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
    else:
        audio_tensor = audio
    with torch.no_grad():
        mel = _GLOBAL_MEL_EXTRACTOR(audio_tensor).squeeze(0).squeeze(0).numpy()
    return mel.astype(np.float32)

def extract_mfcc(audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC):
    """
    Extract 1D aggregated MFCC features with delta and delta-delta.
    Output vector length = n_mfcc * 3 * 2 (e.g., 20*3*2 = 120 features).
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])
    mean_feat = np.mean(combined, axis=1)
    std_feat = np.std(combined, axis=1)

    return np.hstack([mean_feat, std_feat])

def extract_lfcc(audio, sr=config.SAMPLE_RATE, n_lfcc=config.N_LFCC, n_filter=60):
    """
    Extract 1D aggregated Linear Frequency Cepstral Coefficients (LFCC).
    Applies linear filterbank (n_filter) and DCT (n_lfcc), followed by mean & std aggregation.
    """
    stft = librosa.stft(
        audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH
    )
    mag_spec = np.abs(stft)

    fbank = compute_linear_filterbank(sr=sr, n_fft=config.N_FFT, n_filter=n_filter, fmin=config.FMIN, fmax=config.FMAX)
    linear_energies = np.dot(fbank, mag_spec)
    log_energies = np.log(linear_energies + 1e-10)

    lfcc = dct(log_energies, type=2, axis=0, norm='ortho')[:n_lfcc, :]
    delta = librosa.feature.delta(lfcc)
    delta2 = librosa.feature.delta(lfcc, order=2)

    combined = np.vstack([lfcc, delta, delta2])
    mean_feat = np.mean(combined, axis=1)
    std_feat = np.std(combined, axis=1)

    return np.hstack([mean_feat, std_feat])

def extract_spectral_features(audio, sr=config.SAMPLE_RATE):
    """
    Extract classical spectral descriptors (Centroid, Bandwidth, Rolloff, ZCR, RMS).
    """
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=config.HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, hop_length=config.HOP_LENGTH)

    stacked = np.vstack([cent, bw, rolloff, zcr, rms])
    mean_feat = np.mean(stacked, axis=1)
    std_feat = np.std(stacked, axis=1)

    return np.hstack([mean_feat, std_feat])

def extract_mel_spectrogram(audio, sr=config.SAMPLE_RATE, n_mels=config.N_MELS):
    """
    Extract 2D Log-Mel Spectrogram via librosa for legacy routines.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, win_length=config.WIN_LENGTH,
        fmin=config.FMIN, fmax=config.FMAX
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec
