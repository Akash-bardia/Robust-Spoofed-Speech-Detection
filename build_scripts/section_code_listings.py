# -*- coding: utf-8 -*-
"""
Module providing Section 3: Speech Processing Executable Source Code Implementations.
"""

from docx.shared import Pt, Inches, RGBColor

def add_code_listings_section(doc, helpers):
    add_heading_1 = helpers['add_heading_1']
    add_heading_2 = helpers['add_heading_2']
    add_body_p = helpers['add_body_p']
    add_code_block = helpers['add_code_block']

    add_heading_1(doc, "3. Speech Processing & Deep Learning Source Code Implementations")
    add_body_p(doc,
        "This section documents the production-grade Python implementations engineered for the project. Every snippet is fully functional, self-contained, and follows strict PEP 8 standards with comprehensive acoustic commentary for academic defense."
    )

    # LISTING 1: PRE-EMPHASIS & FRAMING
    add_heading_2(doc, "3.1. Speech Pre-Emphasis, Framing, and Windowing")
    add_body_p(doc,
        "Listing 1 implements high-pass pre-emphasis filtering (alpha = 0.97), frame segmentation (25 ms frame size, 10 ms shift), and symmetric Hamming windowing using pure PyTorch tensors for GPU acceleration."
    )
    code_1 = '''import torch
import math

def pre_emphasis_and_frame(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    alpha: float = 0.97
) -> torch.Tensor:
    """
    Applies first-order pre-emphasis filtering and segments speech into windowed frames.
    
    Formula:
        Pre-emphasis: y[n] = x[n] - alpha * x[n-1]
        Hamming window: w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
    
    Args:
        waveform: 1D PyTorch tensor containing audio samples [T]
        sample_rate: Audio sampling frequency (default: 16000 Hz)
        frame_length_ms: Duration of each analysis frame (default: 25.0 ms)
        frame_shift_ms: Hop size between consecutive frames (default: 10.0 ms)
        alpha: Pre-emphasis coefficient (default: 0.97)
    Returns:
        Windowed frames tensor of shape [Num_Frames, Frame_Length_Samples]
    """
    # 1. High-Pass Pre-Emphasis FIR Filter (H(z) = 1 - alpha * z^(-1))
    # Compensates for the -6 dB/octave natural glottal tilt
    padded = torch.cat([waveform[0:1], waveform], dim=0)
    emphasized = padded[1:] - alpha * padded[:-1]

    # 2. Compute integer sample lengths
    frame_len = int(round(sample_rate * frame_length_ms / 1000.0))  # 400 samples
    hop_len = int(round(sample_rate * frame_shift_ms / 1000.0))     # 160 samples

    # 3. Framing via sliding window unfolding
    total_samples = emphasized.shape[0]
    num_frames = max(1, 1 + (total_samples - frame_len) // hop_len)
    
    # Extract overlapping frames [Num_Frames, Frame_Len]
    frames = emphasized.unfold(dimension=0, size=frame_len, step=hop_len)

    # 4. Multiply with symmetric Hamming Window to suppress sidelobes to -43 dB
    n = torch.arange(0, frame_len, dtype=torch.float32, device=waveform.device)
    hamming_win = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n / (frame_len - 1))
    windowed_frames = frames * hamming_win.unsqueeze(0)

    return windowed_frames
'''
    add_code_block(doc, "Listing 1: Speech Pre-Emphasis, Framing, and Windowing (PyTorch)", code_1)

    # LISTING 2: LFCC EXTRACTION
    add_heading_2(doc, "3.2. Linear Frequency Filterbank (LFCC) Extraction with Deltas")
    add_body_p(doc,
        "Listing 2 provides the complete extraction of 60-dimensional Linear Frequency Cepstral Coefficients (LFCC: 20 static + 20 Delta + 20 Delta-Delta) leveraging torchaudio's C++ acceleration engine."
    )
    code_2 = '''import torch
import torchaudio
import torchaudio.functional as AF

class LFCCExtractor(torch.nn.Module):
    """
    High-Resolution Linear Frequency Filterbank (LFCC) Feature Extractor.
    Extracts 60-D spectro-temporal representations (20 static + 20 delta + 20 delta-delta)
    with equal frequency resolution across the 0-8000 Hz Nyquist range.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 60,
        n_lfcc: int = 20,
        n_fft: int = 512,
        win_len: int = 400,
        hop_len: int = 160
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        
        # 1. Construct M=60 Linearly Spaced Triangular Filterbanks
        self.register_buffer("fb", AF.linear_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_filter=n_filter,
            sample_rate=sample_rate
        ))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: waveform of shape [Batch, Time] or [1, Time]
        Output: LFCC tensor of shape [Batch, 1, 60, Time_Frames]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # 2. STFT Magnitude Spectrum Computation
        window = torch.hamming_window(self.win_len, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=window,
            return_complex=True
        )
        power_spec = stft.abs().pow(2.0)  # [Batch, n_freqs, Time]

        # 3. Triangular Filterbank Integration & Logarithmic Compression
        # power_spec [B, F, T] x fb [F, M] -> [B, M, T]
        filter_energies = torch.matmul(power_spec.transpose(-1, -2), self.fb).transpose(-1, -2)
        log_energies = torch.log(filter_energies + 1e-8)

        # 4. Discrete Cosine Transform (DCT-II)
        dct_mat = self._create_dct_matrix(self.fb.shape[1], self.n_lfcc, waveform.device)
        static_lfcc = torch.matmul(dct_mat, log_energies)  # [Batch, 20, Time]

        # 5. Dynamic Derivatives: Velocity (Delta) and Acceleration (Delta-Delta)
        delta_1 = AF.compute_deltas(static_lfcc, win_length=5)
        delta_2 = AF.compute_deltas(delta_1, win_length=5)

        # Stack into complete 60-D feature map [Batch, 1, 60, Time]
        lfcc_60d = torch.cat([static_lfcc, delta_1, delta_2], dim=1)
        return lfcc_60d.unsqueeze(1)

    @staticmethod
    def _create_dct_matrix(n_filters: int, n_lfcc: int, device: torch.device) -> torch.Tensor:
        n = torch.arange(n_lfcc, dtype=torch.float32, device=device).unsqueeze(1)
        k = (torch.arange(n_filters, dtype=torch.float32, device=device) + 0.5).unsqueeze(0)
        dct = torch.cos(torch.pi * n * k / n_filters)
        return dct
'''
    add_code_block(doc, "Listing 2: 60-D Linear Frequency Filterbank (LFCC) Extractor (PyTorch/torchaudio)", code_2)

    # LISTING 3: LPC & LEVINSON-DURBIN
    add_heading_2(doc, "3.3. Linear Predictive Coding (LPC) and Vocal Tract Residual Extraction")
    add_body_p(doc,
        "Listing 3 implements the Levinson-Durbin recursion algorithm to solve the Yule-Walker equations, extracting all-pole vocal tract coefficients a_k and computing the forensic LP excitation residual e(n)."
    )
    code_3 = '''import numpy as np
from scipy.signal import lfilter

def levinson_durbin_lpc(signal: np.ndarray, order: int = 16):
    """
    Computes Linear Predictive Coding (LPC) coefficients and LP prediction residual
    via the Levinson-Durbin recursion algorithm.
    
    Yule-Walker System:
        Sum_{k=1}^p [ a_k * R(|i - k|) ] = -R(i), i = 1, ..., p
    Prediction Error:
        e[n] = s[n] - Sum_{k=1}^p a_k * s[n - k]
    """
    # 1. Compute biased sample autocorrelation R(0), R(1), ..., R(order)
    N = len(signal)
    r = np.correlate(signal, signal, mode='full')[N - 1: N + order]

    # Handle silent frames cleanly
    if r[0] < 1e-12:
        return np.zeros(order), signal, 0.0

    # 2. Levinson-Durbin Recursive Algorithm
    a = np.zeros(order + 1)
    a[0] = 1.0
    alpha = r[0]
    reflection_coeffs = []

    for i in range(1, order + 1):
        # Compute reflection coefficient k_i
        sum_val = np.dot(a[:i], r[1:i + 1][::-1])
        k = (r[i] - sum_val) / alpha
        reflection_coeffs.append(k)

        # Update predictor polynomial
        a_prev = a.copy()
        a[1:i] = a_prev[1:i] - k * a_prev[1:i][::-1]
        a[i] = -k

        # Update prediction error energy
        alpha = alpha * (1.0 - k ** 2)
        if alpha <= 0.0:
            break

    lpc_coeffs = a[1:order + 1]

    # 3. Inverse Filtering to Extract Vocal Tract Excitation Residual e[n]
    # Filter transfer function A(z) = 1 - Sum_{k=1}^p a_k * z^(-k)
    inv_filter_num = np.concatenate([[1.0], -lpc_coeffs])
    inv_filter_den = [1.0]
    residual = lfilter(inv_filter_num, inv_filter_den, signal)

    return lpc_coeffs, residual, alpha
'''
    add_code_block(doc, "Listing 3: Linear Predictive Coding (LPC) & Levinson-Durbin Residual Extraction", code_3)

    # LISTING 4: PITCH TRACKING & HNR
    add_heading_2(doc, "3.4. Pitch (F0) Tracking and Harmonics-to-Noise Ratio (HNR)")
    add_body_p(doc,
        "Listing 4 calculates fundamental frequency F0 and Harmonics-to-Noise Ratio (HNR) via the short-time autocorrelation method to detect prosodic step discontinuities characteristic of synthetic speech."
    )
    code_4 = '''import numpy as np

def compute_pitch_and_hnr(
    frame: np.ndarray,
    sample_rate: int = 16000,
    f0_min: float = 75.0,
    f0_max: float = 400.0
):
    """
    Computes Fundamental Frequency (F0) and Harmonics-to-Noise Ratio (HNR)
    using the short-time autocorrelation function R_x(tau).
    
    HNR formula:
        HNR = 10 * log10( R_x(T0) / (R_x(0) - R_x(T0)) )
    """
    # 1. Compute lags corresponding to pitch range [f0_min, f0_max]
    tau_min = int(sample_rate / f0_max)  # ~40 samples @ 16 kHz
    tau_max = int(sample_rate / f0_min)  # ~213 samples @ 16 kHz

    # 2. Compute autocorrelation
    r = np.correlate(frame, frame, mode='full')
    center = len(frame) - 1
    r_pos = r[center: center + tau_max + 1]

    r_zero = r_pos[0]
    if r_zero < 1e-12:
        return 0.0, -100.0, False  # Silence frame

    # 3. Find primary peak within valid pitch lag window
    search_window = r_pos[tau_min: tau_max + 1]
    peak_offset = np.argmax(search_window)
    t0 = tau_min + peak_offset
    r_t0 = search_window[peak_offset]

    # Voicing decision threshold: normalized autocorrelation r(T0)/r(0) >= 0.35
    norm_ac = r_t0 / r_zero
    is_voiced = norm_ac >= 0.35

    if is_voiced:
        f0 = sample_rate / t0
        # 4. Harmonics-to-Noise Ratio in dB
        noise_energy = max(1e-12, r_zero - r_t0)
        hnr_db = 10.0 * np.log10(max(1e-12, r_t0) / noise_energy)
    else:
        f0 = 0.0
        hnr_db = 0.0

    return f0, hnr_db, is_voiced
'''
    add_code_block(doc, "Listing 4: Pitch (F0) Extraction and Harmonics-to-Noise Ratio (HNR) Analyzer", code_4)

    # LISTING 5: MFM ACTIVATION & DUAL-STREAM LCNN
    add_heading_2(doc, "3.5. Dual-Stream Light CNN (MFM-LCNN) Neural Architecture")
    add_body_p(doc,
        "Listing 5 defines the Max-Feature-Map (MFM) non-linear activation and the academic Dual-Stream LCNN backbone fusing LFCC and Log-Mel feature branches."
    )
    code_5 = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map (MFM) 2D Activation Function.
    Splits channel dimension into two equal halves and takes the element-wise maximum.
    Acts as an adaptive noise filter and feature selector for spectral representations:
        MFM(x) = max(x_{1:C/2}, x_{C/2+1:C})
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, out2 = torch.chunk(x, chunks=2, dim=1)
        return torch.max(out1, out2)

class LCNNEncoder(nn.Module):
    """
    4-Stage Light CNN Encoder with Max-Feature-Map (MFM) non-linearities.
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        # Stage 1: Conv(1 -> 32) -> MFM(16) -> MaxPool(2x2)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MaxFeatureMap2D()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stage 2: Conv(16 -> 48) -> MFM(24) -> MaxPool(2x2)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3, stride=1, padding=1)
        self.mfm2 = MaxFeatureMap2D()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stage 3: Conv(24 -> 64) -> MFM(32) -> MaxPool(2x2)
        self.conv3 = nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1)
        self.mfm3 = MaxFeatureMap2D()
        self.pool3 = nn.MaxPool2d(2, 2)

        # Stage 4: Conv(32 -> 64) -> MFM(32) -> Global Adaptive Average Pooling
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm4 = MaxFeatureMap2D()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool1(self.mfm1(self.conv1(x)))
        x = self.pool2(self.mfm2(self.conv2(x)))
        x = self.pool3(self.mfm3(self.conv3(x)))
        x = self.gap(self.mfm4(self.conv4(x)))
        return x.flatten(1)  # 32-D representation

class DualStreamLCNN(nn.Module):
    """
    Academic State-of-the-Art Dual-Stream MFM-LCNN for ASVspoof 2019 LA:
    • Stream 1: High-resolution LFCC (60 bands) exposing vocoder/synthesis cues.
    • Stream 2: Log-Mel Spectrogram (80 bands) capturing vocal tract formants.
    """
    def __init__(self):
        super().__init__()
        self.lfcc_branch = LCNNEncoder(in_channels=1)
        self.mel_branch = LCNNEncoder(in_channels=1)
        self.classifier = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.35),
            nn.Linear(64, 1)
        )

    def forward(self, lfcc: torch.Tensor, mel: torch.Tensor = None) -> torch.Tensor:
        if mel is None and isinstance(lfcc, (tuple, list)):
            lfcc, mel = lfcc[0], lfcc[1]
        f_lfcc = self.lfcc_branch(lfcc)
        f_mel = self.mel_branch(mel) if mel is not None else f_lfcc
        fused = torch.cat([f_lfcc, f_mel], dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(-1)
'''
    add_code_block(doc, "Listing 5: Dual-Stream MFM-LCNN Architecture (PyTorch)", code_5)

    # LISTING 6: ACOUSTIC HARDENING & SPECAUGMENT
    add_heading_2(doc, "3.6. Acoustic Degradation Hardening and 2D SpecAugment Pipeline")
    add_body_p(doc,
        "Listing 6 implements real-time audio augmentation: Room Impulse Response (RIR) reverberation, calibrated additive Gaussian noise (SNR 10–25 dB), and 2D SpecAugment (time and frequency masking)."
    )
    code_6 = '''import torch
import random
import torchaudio.functional as AF

class AcousticHardeningAugmenter:
    """
    Dual-Domain Data Augmentation Engine:
    1. Time-Domain: Additive Gaussian Noise (SNR 10-25 dB) & Synthetic RIR Reverberation.
    2. Spectrogram-Domain: SpecAugment Frequency & Time Masking.
    """
    def __init__(self, sample_rate: int = 16000, p: float = 0.5):
        self.sample_rate = sample_rate
        self.p = p

    def apply_time_domain(self, waveform: torch.Tensor) -> torch.Tensor:
        """Applies probabilistic time-domain acoustic corruptions."""
        if random.random() > self.p:
            return waveform

        aug_type = random.choice(["noise", "reverb", "lowpass"])

        if aug_type == "noise":
            # Additive white Gaussian noise with calibrated SNR
            snr_db = random.uniform(10.0, 25.0)
            sig_power = waveform.pow(2).mean()
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
            return waveform + noise

        elif aug_type == "reverb":
            # Synthetic exponential decay Room Impulse Response (T60 ~ 0.3s)
            decay_time = random.uniform(0.15, 0.40)
            num_taps = int(self.sample_rate * decay_time)
            t = torch.linspace(0, decay_time, num_taps, device=waveform.device)
            rir = torch.randn(num_taps, device=waveform.device) * torch.exp(-3.0 * t / decay_time)
            rir = rir / rir.norm(p=2)  # Energy normalization
            
            # Convolution via FFT
            wav_len = waveform.shape[-1]
            n_fft = 2 ** (wav_len + num_taps - 1).bit_length()
            w_fft = torch.fft.rfft(waveform, n=n_fft)
            r_fft = torch.fft.rfft(rir, n=n_fft)
            reverbed = torch.fft.irfft(w_fft * r_fft, n=n_fft)[..., :wav_len]
            return reverbed / (reverbed.abs().max() + 1e-8) * waveform.abs().max()

        elif aug_type == "lowpass":
            # Sinc-based lowpass filtering to simulate telephony/microphone rolloff
            cutoff = random.uniform(3000.0, 7000.0)
            return AF.lowpass_biquad(waveform, sample_rate=self.sample_rate, cutoff_freq=cutoff)

        return waveform

    @staticmethod
    def apply_spec_augment(spec: torch.Tensor, freq_mask_param: int = 8, time_mask_param: int = 16):
        """
        Applies 2D SpecAugment (frequency and time band zero-masking).
        Input spec shape: [Batch, Channels, Freq_Bins, Time_Frames]
        """
        cloned = spec.clone()
        # Frequency masking
        f_len = random.randint(0, freq_mask_param)
        f_start = random.randint(0, max(0, spec.shape[2] - f_len))
        cloned[:, :, f_start: f_start + f_len, :] = 0.0

        # Time masking
        t_len = random.randint(0, time_mask_param)
        t_start = random.randint(0, max(0, spec.shape[3] - t_len))
        cloned[:, :, :, t_start: t_start + t_len] = 0.0

        return cloned
'''
    add_code_block(doc, "Listing 6: Acoustic Hardening & SpecAugment Engine (PyTorch)", code_6)

    # LISTING 7: BIOMETRIC EVALUATION ENGINE
    add_heading_2(doc, "3.7. Biometric Evaluation Engine: EER and ROC-AUC Computation")
    add_body_p(doc,
        "Listing 7 provides the standard biometric scoring suite calculating Equal Error Rate (EER), ROC-AUC, Precision, Recall, and F1 under extreme class imbalance."
    )
    code_7 = '''import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support

def compute_biometric_metrics(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Calculates primary biometric performance metrics for speech anti-spoofing:
    • Equal Error Rate (EER): Operating threshold where FAR equals FRR.
    • ROC-AUC: Area under the Receiver Operating Characteristic curve.
    • Optimal Threshold, Precision, Recall, and F1.
    """
    # 1. Receiver Operating Characteristic
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    roc_auc = auc(fpr, tpr)

    # 2. Equal Error Rate (EER) Computation
    # Locate operating point where False Acceptance Rate equals False Rejection Rate
    idx_eer = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2.0
    eer_threshold = thresholds[idx_eer]

    # 3. Decision-Level Classification Metrics
    y_pred = (y_scores >= eer_threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    accuracy = np.mean(y_true == y_pred)

    return {
        "eer": float(eer),
        "eer_percent": float(eer * 100.0),
        "roc_auc": float(roc_auc),
        "eer_threshold": float(eer_threshold),
        "accuracy": float(accuracy * 100.0),
        "precision": float(prec * 100.0),
        "recall": float(rec * 100.0),
        "f1": float(f1 * 100.0)
    }
'''
    add_code_block(doc, "Listing 7: Biometric Evaluation Engine: EER & ROC-AUC Suite", code_7)

    print("Section 3: Code Listings added successfully.")
