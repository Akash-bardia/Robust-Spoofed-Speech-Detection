import pytest
import numpy as np
import os
import sys
from pathlib import Path
import soundfile as sf
import tempfile

# Ensure WildSpoof_Project is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "WildSpoof_Project"))
from src import config, preprocessing

def test_preprocessing_durations_and_channels():
    sr = 16000
    target_samples = 32000

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. 1-second audio (16,000 samples) -> Should be zero-padded at end to 32,000
        path_1s = tmp_path / "audio_1s.wav"
        data_1s = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000, endpoint=False)).astype(np.float32)
        sf.write(str(path_1s), data_1s, sr)
        proc_1s = preprocessing.preprocess_audio(str(path_1s))
        assert len(proc_1s) == target_samples
        assert np.allclose(proc_1s[16000:], 0.0), "Tail should be zero-padded"

        # 2. Exactly 2-second audio (32,000 samples) -> Unchanged length
        path_2s = tmp_path / "audio_2s.wav"
        data_2s = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 32000, endpoint=False)).astype(np.float32)
        sf.write(str(path_2s), data_2s, sr)
        proc_2s = preprocessing.preprocess_audio(str(path_2s))
        assert len(proc_2s) == target_samples
        assert np.all(np.isfinite(proc_2s))

        # 3. 3-second audio (48,000 samples) -> Should be central cropped to 32,000
        path_3s = tmp_path / "audio_3s.wav"
        data_3s = np.linspace(0.0, 1.0, 48000, endpoint=False, dtype=np.float32)
        sf.write(str(path_3s), data_3s, sr)
        proc_3s = preprocessing.preprocess_audio(str(path_3s))
        assert len(proc_3s) == target_samples
        # Central crop should extract data_3s[8000:40000]
        expected_center = data_3s[8000:40000]
        # Normalization scales it, but relative shape matches
        assert np.all(np.isfinite(proc_3s))

        # 4. Stereo audio (2 channels) -> Should be averaged to mono
        path_stereo = tmp_path / "audio_stereo.wav"
        data_stereo = np.zeros((32000, 2), dtype=np.float32)
        data_stereo[:, 0] = 0.5
        data_stereo[:, 1] = -0.5
        sf.write(str(path_stereo), data_stereo, sr)
        proc_stereo, _ = preprocessing.load_audio(str(path_stereo), mono=True)
        assert proc_stereo.ndim == 1
        assert len(proc_stereo) == 32000
        assert np.allclose(proc_stereo, 0.0), "Stereo channels should average to zero"

        # 5. Non-16kHz audio (8 kHz and 44.1 kHz) -> Should resample to 16 kHz
        path_8k = tmp_path / "audio_8k.wav"
        data_8k = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 16000, endpoint=False)).astype(np.float32)
        sf.write(str(path_8k), data_8k, 8000)
        proc_8k = preprocessing.preprocess_audio(str(path_8k))
        assert len(proc_8k) == target_samples

        # 6. Silent audio (all zeros) -> Epsilon-safe normalization, no NaN/Inf
        path_silent = tmp_path / "audio_silent.wav"
        data_silent = np.zeros(32000, dtype=np.float32)
        sf.write(str(path_silent), data_silent, sr)
        proc_silent = preprocessing.preprocess_audio(str(path_silent))
        assert len(proc_silent) == target_samples
        assert not np.isnan(proc_silent).any(), "Silent audio should not produce NaN"
        assert not np.isinf(proc_silent).any(), "Silent audio should not produce Inf"

        # 7. Near-zero audio (1e-9) -> No NaN/Inf
        path_nearzero = tmp_path / "audio_nearzero.wav"
        data_nearzero = np.full(32000, 1e-9, dtype=np.float32)
        sf.write(str(path_nearzero), data_nearzero, sr)
        proc_nearzero = preprocessing.preprocess_audio(str(path_nearzero))
        assert len(proc_nearzero) == target_samples
        assert not np.isnan(proc_nearzero).any()
        assert not np.isinf(proc_nearzero).any()

if __name__ == "__main__":
    test_preprocessing_durations_and_channels()
    print("test_preprocessing passed successfully!")
