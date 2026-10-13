import torch
from torch.utils.data import Dataset
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config

class SpoofDataset(Dataset):
    """
    PyTorch Dataset wrapper for precomputed audio features (Mel Spectrograms, MFCCs, LFCCs).
    """
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

def load_cached_data(split_name="train"):
    cache_path = config.PROCESSED_DIR / f"{split_name}_features.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached features not found: {cache_path}. Run extract_features.py first.")
    
    data = np.load(cache_path)
    return data["X_mfcc"], data["X_lfcc"], data["X_mel"], data["y"]
