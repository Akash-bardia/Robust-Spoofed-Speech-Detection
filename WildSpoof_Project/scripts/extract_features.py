import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, preprocessing, features

def plot_exploratory_analysis():
    print("Generating exploratory speech analysis figures...")
    manifest_path = config.MANIFESTS_DIR / "train.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    real_sample = df[df["label_name"] == "bonafide"].iloc[0]["filepath"]
    fake_sample = df[df["label_name"] == "spoof"].iloc[0]["filepath"]

    audio_real = preprocessing.preprocess_audio(real_sample)
    audio_fake = preprocessing.preprocess_audio(fake_sample)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1. Waveforms
    t = np.linspace(0, config.AUDIO_DURATION, len(audio_real))
    axes[0, 0].plot(t, audio_real, color='navy', alpha=0.8)
    axes[0, 0].set_title("Real Speech Waveform", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(t, audio_fake, color='crimson', alpha=0.8)
    axes[0, 1].set_title("Synthetic Speech Waveform", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Amplitude")

    # 2. Spectrograms
    spec_real = np.abs(librosa.stft(audio_real, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
    spec_fake = np.abs(librosa.stft(audio_fake, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))

    img1 = librosa.display.specshow(librosa.amplitude_to_db(spec_real, ref=np.max),
                                     sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                                     x_axis='time', y_axis='linear', ax=axes[1, 0], cmap='magma')
    axes[1, 0].set_title("Real Speech Spectrogram", fontsize=12, fontweight='bold')
    fig.colorbar(img1, ax=axes[1, 0], format="%+2.0f dB")

    img2 = librosa.display.specshow(librosa.amplitude_to_db(spec_fake, ref=np.max),
                                     sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                                     x_axis='time', y_axis='linear', ax=axes[1, 1], cmap='magma')
    axes[1, 1].set_title("Synthetic Speech Spectrogram", fontsize=12, fontweight='bold')
    fig.colorbar(img2, ax=axes[1, 1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "real_vs_fake_waveform_spectrogram.png", dpi=300)
    plt.close()

    # 3. Mel Spectrogram, MFCC, and LFCC Comparison
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Mel
    mel_real = features.extract_mel_spectrogram(audio_real)
    mel_fake = features.extract_mel_spectrogram(audio_fake)

    librosa.display.specshow(mel_real, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                             x_axis='time', y_axis='mel', fmax=config.FMAX, ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title("Real Speech Mel Spectrogram", fontsize=12, fontweight='bold')

    librosa.display.specshow(mel_fake, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                             x_axis='time', y_axis='mel', fmax=config.FMAX, ax=axes[0, 1], cmap='viridis')
    axes[0, 1].set_title("Synthetic Speech Mel Spectrogram", fontsize=12, fontweight='bold')

    # MFCC
    mfcc_real = librosa.feature.mfcc(y=audio_real, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC)
    mfcc_fake = librosa.feature.mfcc(y=audio_fake, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC)

    librosa.display.specshow(mfcc_real, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                             x_axis='time', ax=axes[1, 0], cmap='coolwarm')
    axes[1, 0].set_title("Real Speech MFCC", fontsize=12, fontweight='bold')

    librosa.display.specshow(mfcc_fake, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                             x_axis='time', ax=axes[1, 1], cmap='coolwarm')
    axes[1, 1].set_title("Synthetic Speech MFCC", fontsize=12, fontweight='bold')

    # LFCC
    stft_real = librosa.stft(audio_real, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    stft_fake = librosa.stft(audio_fake, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    fbank = features.compute_linear_filterbank()
    lfcc_real = features.dct(np.log(np.dot(fbank, np.abs(stft_real)) + 1e-10), type=2, axis=0, norm='ortho')[:config.N_LFCC, :]
    lfcc_fake = features.dct(np.log(np.dot(fbank, np.abs(stft_fake)) + 1e-10), type=2, axis=0, norm='ortho')[:config.N_LFCC, :]

    librosa.display.specshow(lfcc_real, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                             x_axis='time', ax=axes[2, 0], cmap='inferno')
    axes[2, 0].set_title("Real Speech LFCC", fontsize=12, fontweight='bold')

    librosa.display.specshow(lfcc_fake, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                             x_axis='time', ax=axes[2, 1], cmap='inferno')
    axes[2, 1].set_title("Synthetic Speech LFCC", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "real_vs_fake_mfcc_lfcc_mel.png", dpi=300)
    plt.close()

    print("Exploratory speech visualisations saved successfully!")

def extract_and_cache_features(split_name="train", max_samples=None):
    manifest_path = config.MANIFESTS_DIR / f"{split_name}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if max_samples is not None and max_samples < len(df):
        # Sample per class for balance
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), max_samples // 2), random_state=config.RANDOM_SEED))

    print(f"Extracting features for {split_name} ({len(df)} files)...")

    mfcc_list, lfcc_list, mel_list, label_list = [], [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        filepath = row["filepath"]
        label = row["label"]

        audio = preprocessing.preprocess_audio(filepath)

        mfcc_feat = features.extract_mfcc(audio)
        lfcc_feat = features.extract_lfcc(audio)
        mel_feat = features.extract_mel_spectrogram(audio)

        mfcc_list.append(mfcc_feat)
        lfcc_list.append(lfcc_feat)
        mel_list.append(mel_feat)
        label_list.append(label)

    X_mfcc = np.array(mfcc_list)
    X_lfcc = np.array(lfcc_list)
    X_mel = np.array(mel_list)  # (N, n_mels, time_steps)
    y = np.array(label_list)

    cache_path = config.PROCESSED_DIR / f"{split_name}_features.npz"
    np.savez_compressed(cache_path, X_mfcc=X_mfcc, X_lfcc=X_lfcc, X_mel=X_mel, y=y)
    print(f"Cached {split_name} features to {cache_path}: MFCC shape {X_mfcc.shape}, LFCC shape {X_lfcc.shape}, Mel shape {X_mel.shape}")

    return X_mfcc, X_lfcc, X_mel, y

if __name__ == "__main__":
    plot_exploratory_analysis()
    extract_and_cache_features("train", config.MAX_TRAIN_SAMPLES)
    extract_and_cache_features("validation", config.MAX_VAL_SAMPLES)
    extract_and_cache_features("test", config.MAX_TEST_SAMPLES)
