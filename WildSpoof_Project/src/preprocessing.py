import numpy as np
import librosa
import soundfile as sf
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

def load_audio(filepath, target_sr=config.SAMPLE_RATE, mono=config.MONO):
    """
    Load an audio file, converting to mono if requested.
    Handles corrupt/invalid files gracefully by returning zeros.
    """
    try:
        audio, sr = sf.read(filepath, dtype='float32')
        if audio.ndim > 1 and mono:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)
        return audio, target_sr
    except Exception as e:
        print(f"Warning: Error loading {filepath} ({e}). Returning zero signal.")
        num_samples = int(target_sr * config.AUDIO_DURATION)
        return np.zeros(num_samples, dtype=np.float32), target_sr

def resample_audio(audio, orig_sr, target_sr=config.SAMPLE_RATE):
    """
    Resample audio signal from orig_sr to target_sr using librosa.
    """
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def normalize_audio(audio):
    """
    Normalize peak amplitude of audio to [-1.0, 1.0].
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

def pad_or_crop(audio, target_samples=config.NUM_SAMPLES):
    """
    Pad with zeros or crop signal to exact target length.
    """
    num_samples = len(audio)
    if num_samples == target_samples:
        return audio
    elif num_samples > target_samples:
        # Crop from center or start
        return audio[:target_samples]
    else:
        # Pad with zeros
        pad_width = target_samples - num_samples
        return np.pad(audio, (0, pad_width), mode='constant')

def preprocess_audio(filepath, target_sr=config.SAMPLE_RATE, target_duration=config.AUDIO_DURATION):
    """
    Complete audio preprocessing pipeline:
    1. Load audio
    2. Convert to mono
    3. Resample to target_sr (16 kHz)
    4. Peak amplitude normalization
    5. Pad or crop to target_duration (2.0s)
    """
    target_samples = int(target_sr * target_duration)
    audio, sr = load_audio(filepath, target_sr=target_sr, mono=True)
    if config.NORMALIZE_AMPLITUDE:
        audio = normalize_audio(audio)
    audio = pad_or_crop(audio, target_samples=target_samples)
    return audio
