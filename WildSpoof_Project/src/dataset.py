"""
Canonical dataset entry point for ASVspoof 2019 Logical Access (LA).
Exports ASVspoofLADataset and backward-compatible SpoofDataset alias.
"""
from .la_dataset import ASVspoofLADataset

# Backward-compatible alias
SpoofDataset = ASVspoofLADataset

__all__ = ["ASVspoofLADataset", "SpoofDataset"]
