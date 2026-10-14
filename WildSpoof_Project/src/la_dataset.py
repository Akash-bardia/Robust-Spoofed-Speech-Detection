import os
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config, augmentation, preprocessing, features

# Global feature extractors for maximum reproducibility and throughput
_LFCC_EXTRACTOR = features.LFCCExtractor(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    win_length=config.WIN_LENGTH,
    hop_length=config.HOP_LENGTH,
    n_filter=config.N_LFCC_FILTERS,
    n_lfcc=config.N_LFCC,
    f_min=config.FMIN,
    f_max=config.FMAX
)

_MEL_EXTRACTOR = features.LogMelExtractor(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    win_length=config.WIN_LENGTH,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS,
    f_min=config.FMIN,
    f_max=config.FMAX
)

# Aliases for backward compatibility
_LFCC_TRANSFORM = _LFCC_EXTRACTOR.lfcc_transform
_MEL_TRANSFORM = _MEL_EXTRACTOR.mel_transform
_AMP_TO_DB = _MEL_EXTRACTOR.amp_to_db

def load_and_standardize_audio(filepath, target_samples=config.NUM_SAMPLES):
    """
    Unified audio loader delegating to preprocessing pipeline:
    - 16 kHz mono
    - Epsilon-safe peak amplitude normalization
    - Central-crop if > target_samples, zero-pad if < target_samples to exactly target_samples (32,000)
    """
    return preprocessing.preprocess_audio(filepath)

def extract_2d_lfcc(audio, sr=config.SAMPLE_RATE):
    """
    Extract 2D LFCC spectrogram tensor of shape (n_lfcc, time_frames).
    """
    return features.extract_2d_lfcc(audio, sr=sr)

def extract_2d_mel(audio, sr=config.SAMPLE_RATE):
    """
    Extract 2D Log-Mel spectrogram tensor of shape (n_mels, time_frames).
    """
    return features.extract_2d_mel(audio, sr=sr)

def apply_specaugment(feat_tensor, max_f=12, max_t=25):
    """
    SpecAugment delegating to augmentation module.
    """
    return augmentation.apply_specaugment(feat_tensor, max_f=max_f, max_t=max_t)

class ASVspoofLADataset(Dataset):
    """
    PyTorch Dataset for ASVspoof 2019 Logical Access speech anti-spoofing.
    Extracts 2D acoustic representations on-the-fly with optional acoustic hardening and SpecAugment.
    Supports feature_type: 'dual' (LFCC + Log-Mel), 'lfcc', or 'mel'.
    """
    def __init__(self, df, feature_type="dual", augment=False, target_samples=config.NUM_SAMPLES):
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

        # 1. Load and standardize audio (exactly 32,000 samples)
        audio = load_and_standardize_audio(filepath, target_samples=self.target_samples)

        # 2. Time-domain acoustic augmentation if enabled (training only)
        if self.augment:
            rand_choice = np.random.rand()
            if rand_choice < 0.30:
                # Additive Gaussian white noise with random SNR (5 to 25 dB)
                snr = np.random.uniform(5.0, 25.0)
                audio = augmentation.add_gaussian_noise(audio, snr_db=snr)
            elif rand_choice < 0.55:
                # Synthetic room impulse response reverberation
                decay = np.random.uniform(10.0, 25.0)
                audio = augmentation.apply_reverberation(audio, decay_rate=decay)
            elif rand_choice < 0.75:
                # Combined additive noise + reverberation
                snr = np.random.uniform(10.0, 20.0)
                noisy = augmentation.add_gaussian_noise(audio, snr_db=snr)
                decay = np.random.uniform(10.0, 20.0)
                audio = augmentation.apply_reverberation(noisy, decay_rate=decay)
            elif rand_choice < 0.90:
                # Lowpass filtering (narrowband telephony simulation)
                cutoff = np.random.uniform(2800.0, 3800.0)
                audio = augmentation.apply_lowpass_filter(audio, cutoff_freq=cutoff)

        w_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, 32000)

        # 3. Extract acoustic representations
        with torch.no_grad():
            if self.feature_type == "dual":
                lfcc = _LFCC_EXTRACTOR(w_t)  # (1, 1, 60, 201)
                mel = _MEL_EXTRACTOR(w_t)    # (1, 1, 80, 201)
                if self.augment:
                    lfcc = augmentation.apply_specaugment(lfcc, max_f=12, max_t=25)
                    mel = augmentation.apply_specaugment(mel, max_f=16, max_t=25)
                # Squeeze to (1, F, T) for LCNN input
                feat = (lfcc.squeeze(0), mel.squeeze(0))
            elif self.feature_type == "lfcc":
                lfcc = _LFCC_EXTRACTOR(w_t)
                if self.augment:
                    lfcc = augmentation.apply_specaugment(lfcc, max_f=12, max_t=25)
                feat = lfcc.squeeze(0)
            else:  # mel
                mel = _MEL_EXTRACTOR(w_t)
                if self.augment:
                    mel = augmentation.apply_specaugment(mel, max_f=16, max_t=25)
                feat = mel.squeeze(0)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return feat, label_tensor, row["audio_id"], row["system_id"]
