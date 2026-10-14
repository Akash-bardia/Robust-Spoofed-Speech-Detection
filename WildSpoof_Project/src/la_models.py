import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFM(nn.Module):
    """
    Max-Feature-Map (MFM) activation function.
    Splits the channel dimension into two equal halves (C/2) and takes the element-wise maximum:
      MFM(X) = max(X1, X2)
    Acts as a competitive non-linear feature filter that suppresses noise while propagating
    subtle synthetic speech cues (Wu et al., IEEE TIFS 2018).
    Note: MFM is a max-out activation, NOT ReLU.
    """
    def forward(self, x):
        channels = x.shape[1]
        assert channels % 2 == 0, f"MFM requires an even number of channels, got {channels}"
        out = torch.split(x, channels // 2, dim=1)
        return torch.max(out[0], out[1])

class LCNNEncoder(nn.Module):
    """
    4-Stage Convolutional Encoder with Max-Feature-Map (MFM) non-linearities.
    Designed for fine-grained spectro-temporal artifact extraction in speech anti-spoofing.
    Stage 1: Conv2d(in_channels -> 32) -> MFM(16) -> MaxPool2d(2, 2)
    Stage 2: Conv2d(16 -> 48) -> MFM(24) -> MaxPool2d(2, 2)
    Stage 3: Conv2d(24 -> 64) -> MFM(32) -> MaxPool2d(2, 2)
    Stage 4: Conv2d(32 -> 2 * out_dim) -> MFM(out_dim) -> AdaptiveAvgPool2d((1, 1))
    Output: Flattened embedding tensor of shape (B, out_dim). Default out_dim = 64.
    """
    def __init__(self, in_channels=1, out_dim=64):
        super(LCNNEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Stage 1: Conv(1 -> 32) -> MFM(16) -> MaxPool
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MFM()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: Conv(16 -> 48) -> MFM(24) -> MaxPool
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3, stride=1, padding=1)
        self.mfm2 = MFM()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: Conv(24 -> 64) -> MFM(32) -> MaxPool
        self.conv3 = nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1)
        self.mfm3 = MFM()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 4: Conv(32 -> 2 * out_dim) -> MFM(out_dim) -> Global Adaptive Average Pooling
        self.conv4 = nn.Conv2d(32, 2 * out_dim, kernel_size=3, stride=1, padding=1)
        self.mfm4 = MFM()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, F, T)

        x = self.pool1(self.mfm1(self.conv1(x)))
        x = self.pool2(self.mfm2(self.conv2(x)))
        x = self.pool3(self.mfm3(self.conv3(x)))
        x = self.gap(self.mfm4(self.conv4(x)))
        out = x.flatten(1)
        assert out.shape[1] == self.out_dim, f"Expected {self.out_dim}-D output, got {out.shape[1]}"
        return out

class DualStreamLCNN(nn.Module):
    """
    Dual-Stream LFCC + Log-Mel Light CNN with Max-Feature-Map (MFM) Activations.
    Academic state-of-the-art architecture for ASVspoof 2019 Logical Access:
    - Branch 1: High-resolution LFCC (linear filterbank) capturing vocoder & waveform synthesis artifacts.
    - Branch 2: 2D Log-Mel Spectrogram capturing perceptual harmonic spectral envelope dynamics.
    - Joint Representation: Concatenation of LFCC (out_dim) + Log-Mel (out_dim) -> (2 * out_dim).
      Default canonical: 64-D + 64-D = 128-D fused embedding.
    - Classifier:
        Linear(2 * out_dim -> 64)
        -> BatchNorm1d(64)
        -> LeakyReLU(0.1)
        -> Dropout(dropout)
        -> Linear(64 -> num_classes)
      Where num_classes = 2 (logits for [bona_fide, spoof]) or 1 (binary logit for backward compatibility).
    """
    def __init__(self, embedding_dim=64, num_classes=2, dropout=0.4):
        super(DualStreamLCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.lfcc_branch = LCNNEncoder(in_channels=1, out_dim=embedding_dim)
        self.mel_branch = LCNNEncoder(in_channels=1, out_dim=embedding_dim)

        fused_dim = 2 * embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def extract_embeddings(self, lfcc, mel):
        """Extract separate 64-D embeddings and concatenated 128-D joint embedding."""
        if mel is None and isinstance(lfcc, (tuple, list)):
            lfcc, mel = lfcc[0], lfcc[1]
        f_lfcc = self.lfcc_branch(lfcc)
        f_mel = self.mel_branch(mel)
        fused = torch.cat([f_lfcc, f_mel], dim=1)
        return f_lfcc, f_mel, fused

    def forward(self, lfcc, mel=None):
        if mel is None:
            if isinstance(lfcc, (tuple, list)):
                lfcc, mel = lfcc[0], lfcc[1]
            else:
                # Single stream fallback: duplicate branch
                e = self.lfcc_branch(lfcc)
                fused = torch.cat([e, e], dim=1)
                logits = self.classifier(fused)
                return logits.squeeze(-1) if self.num_classes == 1 else logits

        f_lfcc = self.lfcc_branch(lfcc)
        f_mel = self.mel_branch(mel)
        fused = torch.cat([f_lfcc, f_mel], dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(-1) if self.num_classes == 1 else logits

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device="cpu"):
        """
        Dynamically detects checkpoint configuration (embedding_dim and num_classes)
        to ensure 100% backward compatibility with legacy checkpoints and new models.
        """
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Detect embedding dim from lfcc_branch.conv4.weight shape: [2*embedding_dim, 32, 3, 3]
        conv4_out = state_dict["lfcc_branch.conv4.weight"].shape[0]
        detected_embedding_dim = conv4_out // 2

        # Detect num_classes from classifier.4.weight shape: [num_classes, 64]
        detected_num_classes = state_dict["classifier.4.weight"].shape[0]

        model = cls(embedding_dim=detected_embedding_dim, num_classes=detected_num_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

def get_model_summary(model, input_shapes=((1, 1, 60, 201), (1, 1, 80, 201))):
    """
    Compute rigorous parameter count and memory consumption breakdown:
    - Trainable parameter count
    - Non-trainable parameter count (BatchNorm running stats)
    - FP32 parameter memory
    - FP16 parameter memory
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(b.numel() for b in model.buffers())
    total_params = trainable_params + non_trainable_params

    fp32_bytes = total_params * 4
    fp16_bytes = total_params * 2

    return {
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "total_parameters": total_params,
        "fp32_memory_bytes": fp32_bytes,
        "fp32_memory_kb": fp32_bytes / 1024.0,
        "fp16_memory_bytes": fp16_bytes,
        "fp16_memory_kb": fp16_bytes / 1024.0
    }

class DeepSpoofCNN(nn.Module):
    """
    Legacy single-stream 4-stage CNN for comparison benchmarks.
    """
    def __init__(self, in_channels=1):
        super(DeepSpoofCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.35)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1))
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits.squeeze(-1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.leaky_relu(out, 0.1)

class ResNetSpoofNet(nn.Module):
    """
    Residual Anti-Spoofing Network for spectro-temporal artifact comparison.
    """
    def __init__(self, in_channels=1):
        super(ResNetSpoofNet, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )
        self.res1 = ResidualBlock(32)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.res2 = ResidualBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        self.res3 = ResidualBlock(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.in_conv(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits.squeeze(-1)
