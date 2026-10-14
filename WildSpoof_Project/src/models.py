"""
Canonical model architecture entry point.
Exports DualStreamLCNN, LCNNEncoder, MFM layer, and model summary profiler.
"""
from .la_models import DualStreamLCNN, LCNNEncoder, MFM, get_model_summary

__all__ = ["DualStreamLCNN", "LCNNEncoder", "MFM", "get_model_summary"]
