import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSpoofCNN(nn.Module):
    """
    Enhanced Deep 2D Convolutional Neural Network for ASVspoof 2019 detection.
    Compatible with both 2D LFCC (40 bands) and 2D Log-Mel (80 bands).
    """
    def __init__(self, in_channels=1):
        super(DeepSpoofCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        # Block 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Global average pooling to handle variable time lengths cleanly
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier head
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
    Residual Anti-Spoofing Network for fine-grained spectro-temporal artifact detection.
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

class MFM(nn.Module):
    """
    Max-Feature-Map (MFM) activation function.
    Splits the channel dimension into two equal halves and takes the element-wise maximum.
    Acts as a non-linear feature filter that suppresses noise while propagating synthetic speech cues.
    """
    def forward(self, x):
        out = torch.split(x, x.shape[1] // 2, 1)
        return torch.max(out[0], out[1])

class LCNNEncoder(nn.Module):
    """
    4-stage Convolutional Encoder with Max-Feature-Map (MFM) non-linearities.
    Engineered for fine-grained spectro-temporal artifact extraction in speech anti-spoofing.
    """
    def __init__(self, in_channels=1):
        super(LCNNEncoder, self).__init__()
        # Stage 1: Conv(1 -> 32) -> MFM(16) -> MaxPool
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MFM()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stage 2: Conv(16 -> 48) -> MFM(24) -> MaxPool
        self.conv2 = nn.Conv2d(16, 48, kernel_size=3, stride=1, padding=1)
        self.mfm2 = MFM()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stage 3: Conv(24 -> 64) -> MFM(32) -> MaxPool
        self.conv3 = nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1)
        self.mfm3 = MFM()
        self.pool3 = nn.MaxPool2d(2, 2)

        # Stage 4: Conv(32 -> 64) -> MFM(32) -> Global Adaptive Pooling
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm4 = MFM()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool1(self.mfm1(self.conv1(x)))
        x = self.pool2(self.mfm2(self.conv2(x)))
        x = self.pool3(self.mfm3(self.conv3(x)))
        x = self.gap(self.mfm4(self.conv4(x)))
        return x.flatten(1) # 32-D representation

class DualStreamLCNN(nn.Module):
    """
    Dual-Stream LFCC + Log-Mel Light CNN with Max-Feature-Map (MFM) Activations.
    Academic state-of-the-art architecture for ASVspoof 2019 Logical Access:
    - Branch 1: High-resolution LFCC (linear filterbank) capturing vocoder & waveform synthesis artifacts.
    - Branch 2: 2D Log-Mel Spectrogram capturing perceptual harmonic spectral envelope dynamics.
    - Fused Representation: Joint embedding passed through Batch Normalization, LeakyReLU, and Dropout.
    """
    def __init__(self):
        super(DualStreamLCNN, self).__init__()
        self.lfcc_branch = LCNNEncoder(in_channels=1)
        self.mel_branch = LCNNEncoder(in_channels=1)
        self.classifier = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.35),
            nn.Linear(64, 1)
        )

    def forward(self, lfcc, mel=None):
        if mel is None:
            if isinstance(lfcc, (tuple, list)):
                lfcc, mel = lfcc[0], lfcc[1]
            else:
                # Fallback if only single feature passed
                e = self.lfcc_branch(lfcc)
                return self.classifier(torch.cat([e, e], dim=1)).squeeze(-1)

        f_lfcc = self.lfcc_branch(lfcc)
        f_mel = self.mel_branch(mel)
        fused = torch.cat([f_lfcc, f_mel], dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(-1)

