import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import librosa
from scipy.fftpack import dct
import torchaudio.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config, augmentation

# Global transforms for maximum performance
_LFCC_TRANSFORM = T.LFCC(
    sample_rate=16000, n_filter=128, n_lfcc=60, log_lf=True,
    speckwargs={'n_fft': 512, 'win_length': 400, 'hop_length': 160}
)
_MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80
)
_AMP_TO_DB = T.AmplitudeToDB()

def load_and_standardize_audio(filepath, target_samples=32000):
    """
    Fast FLAC / WAV audio loader.
    Ensures single-channel mono 16kHz, peak-normalization, and pads or trims to target_samples.
    """
    try:
        audio, sr = sf.read(filepath, dtype='float32')
    except Exception:
        audio, sr = librosa.load(filepath, sr=16000)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Peak normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Trim or pad to exact target_samples
    if len(audio) > target_samples:
        start = (len(audio) - target_samples) // 2
        audio = audio[start : start + target_samples]
    elif len(audio) < target_samples:
        pad_width = target_samples - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')

    return audio.astype(np.float32)

def extract_2d_lfcc(audio, sr=16000, n_fft=512, hop_length=160, win_length=400, n_lfcc=40, n_filter=40):
    """
    Extract 2D Linear Frequency Cepstral Coefficients (LFCC) spectrogram.
    Returns array of shape (n_lfcc, time_frames).
    """
    w_t = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        lfcc = _LFCC_TRANSFORM(w_t).squeeze(0).numpy()
    return lfcc.astype(np.float32)

def extract_2d_mel(audio, sr=16000, n_mels=80, n_fft=512, hop_length=160, win_length=400):
    """
    Extract 2D Log-Mel Spectrogram.
    Returns array of shape (n_mels, time_frames).
    """
    w_t = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        mel = _AMP_TO_DB(_MEL_TRANSFORM(w_t)).squeeze(0).numpy()
    return mel.astype(np.float32)

def apply_specaugment(feat_tensor, max_f=12, max_t=25):
    """
    SpecAugment: Randomly masks frequency channels and time frames to boost unseen attack generalization.
    """
    feat = feat_tensor.clone()
    _, f_bins, t_frames = feat.shape
    if np.random.rand() < 0.45:
        f_mask = np.random.randint(4, min(max_f, f_bins // 2))
        f0 = np.random.randint(0, f_bins - f_mask)
        feat[:, f0:f0+f_mask, :] = 0.0

    if np.random.rand() < 0.45:
        t_mask = np.random.randint(8, min(max_t, t_frames // 2))
        t0 = np.random.randint(0, t_frames - t_mask)
        feat[:, :, t0:t0+t_mask] = 0.0

    return feat

class ASVspoofLADataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 Logical Access audio files.
    Extracts 2D acoustic features on the fly with optional acoustic augmentation and SpecAugment.
    Supports feature_type: 'dual' (LFCC + Log-Mel), 'lfcc', or 'mel'.
    """
    def __init__(self, df, feature_type="dual", augment=False, target_samples=32000):
        self.df = df.reset_index(drop=True)
        self.feature_type = feature_type.lower()
        self.augment = augment
        self.target_samples = target_samples

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = int(row["label"])

        # Load audio
        audio = load_and_standardize_audio(filepath, target_samples=self.target_samples)

        # On-the-fly acoustic augmentation if enabled
        if self.augment:
            rand_choice = np.random.rand()
            if rand_choice < 0.30:
                # Additive noise with random SNR (5 to 25 dB)
                snr = np.random.uniform(5.0, 25.0)
                audio = augmentation.add_gaussian_noise(audio, snr_db=snr)
            elif rand_choice < 0.55:
                # Synthetic room reverberation
                decay = np.random.uniform(10.0, 25.0)
                audio = augmentation.apply_reverberation(audio, decay_rate=decay)
            elif rand_choice < 0.75:
                # Combined noise + reverberation
                snr = np.random.uniform(10.0, 20.0)
                noisy = augmentation.add_gaussian_noise(audio, snr_db=snr)
                decay = np.random.uniform(10.0, 20.0)
                audio = augmentation.apply_reverberation(noisy, decay_rate=decay)
            elif rand_choice < 0.90:
                # Lowpass filter
                cutoff = np.random.uniform(2800, 3800)
                audio = augmentation.apply_lowpass_filter(audio, cutoff_freq=cutoff)

        w_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0) # (1, 32000)

        if self.feature_type == "dual":
            lfcc = _LFCC_TRANSFORM(w_t) # (1, 60, 201)
            mel = _AMP_TO_DB(_MEL_TRANSFORM(w_t)) # (1, 80, 201)
            if self.augment:
                lfcc = apply_specaugment(lfcc, max_f=12, max_t=25)
                mel = apply_specaugment(mel, max_f=16, max_t=25)
            feat = (lfcc, mel)
        elif self.feature_type == "lfcc":
            lfcc = _LFCC_TRANSFORM(w_t)
            if self.augment:
                lfcc = apply_specaugment(lfcc, max_f=12, max_t=25)
            feat = lfcc
        else: # mel
            mel = _AMP_TO_DB(_MEL_TRANSFORM(w_t))
            if self.augment:
                mel = apply_specaugment(mel, max_f=16, max_t=25)
            feat = mel

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return feat, label_tensor, row["audio_id"], row["system_id"]

