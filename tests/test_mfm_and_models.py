import pytest
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "WildSpoof_Project"))
from src import config
from src.la_models import MFM, LCNNEncoder, DualStreamLCNN, get_model_summary

def test_mfm_activation():
    mfm = MFM()

    # 1. Element-wise maximum across channel halves
    # Shape: (1, 4, 2, 2)
    # Channel 0: [[1, 2], [3, 4]]
    # Channel 1: [[5, 6], [7, 8]]
    # Channel 2: [[2, 1], [4, 3]]
    # Channel 3: [[4, 7], [6, 9]]
    c0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    c1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    c2 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    c3 = torch.tensor([[4.0, 7.0], [6.0, 9.0]])
    x = torch.stack([c0, c1, c2, c3], dim=0).unsqueeze(0)  # (1, 4, 2, 2)

    out = mfm(x)
    assert out.shape == (1, 2, 2, 2), f"Expected channel dimension to halve from 4 to 2, got {out.shape}"
    # Expected output:
    # Out channel 0: max(c0, c2) -> [[2, 2], [4, 4]]
    # Out channel 1: max(c1, c3) -> [[5, 7], [7, 9]]
    expected_c0 = torch.max(c0, c2)
    expected_c1 = torch.max(c1, c3)
    assert torch.allclose(out[0, 0], expected_c0)
    assert torch.allclose(out[0, 1], expected_c1)

    # 2. Gradient flow test
    x.requires_grad = True
    y = mfm(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))

    # 3. Assert error on odd number of channels
    odd_x = torch.randn(1, 3, 4, 4)
    with pytest.raises(AssertionError):
        mfm(odd_x)

def test_lcnn_encoder_and_dual_stream():
    batch_size = 2
    lfcc_feat = torch.randn(batch_size, 1, 60, 201)
    mel_feat = torch.randn(batch_size, 1, 80, 201)

    # 1. Canonical Dual-Stream LCNN: 64-D per branch -> 128-D fused -> 2 classes
    canon_model = DualStreamLCNN(embedding_dim=64, num_classes=2)
    canon_logits = canon_model(lfcc_feat, mel_feat)
    assert canon_logits.shape == (batch_size, 2), f"Expected logits shape (2, 2), got {canon_logits.shape}"

    f_lfcc, f_mel, fused = canon_model.extract_embeddings(lfcc_feat, mel_feat)
    assert f_lfcc.shape == (batch_size, 64), f"Expected 64-D LFCC embedding, got {f_lfcc.shape}"
    assert f_mel.shape == (batch_size, 64), f"Expected 64-D Mel embedding, got {f_mel.shape}"
    assert fused.shape == (batch_size, 128), f"Expected 128-D fused embedding, got {fused.shape}"

    sum_canon = get_model_summary(canon_model)
    assert sum_canon["trainable_parameters"] == 125858, f"Unexpected parameter count: {sum_canon['trainable_parameters']}"

    # 2. Legacy Checkpoint Architecture: 32-D per branch -> 64-D fused -> 1 class
    legacy_model = DualStreamLCNN(embedding_dim=32, num_classes=1)
    legacy_logits = legacy_model(lfcc_feat, mel_feat)
    assert legacy_logits.shape == (batch_size,), f"Expected shape (2,), got {legacy_logits.shape}"

    sum_legacy = get_model_summary(legacy_model)
    assert sum_legacy["trainable_parameters"] == 84705, f"Unexpected legacy parameter count: {sum_legacy['trainable_parameters']}"

    # 3. Verify checkpoint loading on disk
    ckpt_path = config.MODELS_DIR / "la_academic_lcnn_baseline.pt"
    if ckpt_path.exists():
        loaded_model = DualStreamLCNN.load_from_checkpoint(ckpt_path)
        out_loaded = loaded_model(lfcc_feat, mel_feat)
        assert out_loaded.shape == (batch_size,)
        assert loaded_model.embedding_dim == 32
        assert loaded_model.num_classes == 1

if __name__ == "__main__":
    test_mfm_activation()
    test_lcnn_encoder_and_dual_stream()
    print("test_mfm_and_models passed successfully!")
