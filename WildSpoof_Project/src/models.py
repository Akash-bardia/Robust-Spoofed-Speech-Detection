import torch
import torch.nn as nn
import torch.nn.functional as F

class SpoofCNN(nn.Module):
    """
    Lightweight 2D Convolutional Neural Network for binary speech spoofing detection.
    Input: Mel-Spectrogram tensor of shape (Batch, 1, n_mels, time_steps)
    """
    def __init__(self, in_channels=1, n_mels=80):
        super(SpoofCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)  # Binary logit output

    def forward(self, x):
        # Add channel dimension if 3D (Batch, n_mels, time_steps)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits.squeeze(-1)
