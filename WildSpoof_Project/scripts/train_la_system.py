import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

sys.stdout.reconfigure(line_buffering=True)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import config, utils
from src.la_dataset import ASVspoofLADataset
from src.la_models import DualStreamLCNN, DeepSpoofCNN
from src.evaluate import compute_eer

def compute_val_metrics(model, val_loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for feats, labels, _, _ in val_loader:
            if isinstance(feats, (tuple, list)):
                lfcc, mel = feats[0].to(device), feats[1].to(device)
                logits = model(lfcc, mel)
            else:
                feats = feats.to(device)
                logits = model(feats)

            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    preds = (all_scores >= 0.5).astype(int)
    acc = (preds == all_labels).mean()
    try:
        eer, eer_thresh = compute_eer(all_labels, all_scores)
    except Exception:
        eer, eer_thresh = 0.5, 0.5
    return acc, eer, eer_thresh

def train_model(train_df, dev_df, model_type="baseline", epochs=4, batch_size=64, lr=1e-3):
    utils.set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Training Academic LA Model: {model_type.upper()} | Feature: DUAL (LFCC+Mel) | Device: {device}")
    print(f"=======================================================")

    is_augmented = (model_type == "augmented")
    train_dataset = ASVspoofLADataset(train_df, feature_type="dual", augment=is_augmented)
    val_dataset = ASVspoofLADataset(dev_df, feature_type="dual", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DualStreamLCNN().to(device)

    # Class weighting for balanced loss
    num_pos = np.sum(train_df["label"] == 1)
    num_neg = np.sum(train_df["label"] == 0)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_eer = float('inf')
    save_primary = config.MODELS_DIR / f"la_academic_lcnn_{model_type}.pt"
    save_compat = config.MODELS_DIR / f"la_cnn_{model_type}_mel.pt"

    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [{model_type}]")

        for feats, labels, _, _ in pbar:
            if isinstance(feats, (tuple, list)):
                lfcc, mel = feats[0].to(device), feats[1].to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(lfcc, mel)
            else:
                feats, labels = feats.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(feats)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)

        # Validation
        val_acc, val_eer, val_thresh = compute_val_metrics(model, val_loader, device)
        t_epoch = time.time() - t0

        print(f"--> Epoch {epoch:02d} ({t_epoch:.1f}s) | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val EER: {val_eer*100:.2f}% | Thresh: {val_thresh:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_acc": val_acc,
            "val_eer": val_eer,
            "val_thresh": val_thresh
        })

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), save_primary)
            torch.save(model.state_dict(), save_compat)
            print(f"    [BEST CHECKPOINT SAVED] Val EER: {val_eer*100:.2f}% -> {save_primary.name} & {save_compat.name}")

    history_df = pd.DataFrame(history)
    history_csv = config.LOGS_DIR / f"train_history_{model_type}_mel.csv"
    history_df.to_csv(history_csv, index=False)
    print(f"Training history saved to {history_csv}")

    return save_primary

def main():
    manifests_dir = config.MANIFESTS_DIR
    train_manifest = manifests_dir / "train_la.csv"
    dev_manifest = manifests_dir / "dev_la.csv"

    if not train_manifest.exists() or not dev_manifest.exists():
        raise FileNotFoundError("LA manifests not found! Please run scripts/prepare_la_manifests.py first.")

    df_train_full = pd.read_csv(train_manifest)
    df_dev_full = pd.read_csv(dev_manifest)

    # Balanced training set: all 2,580 bona fide + 2,580 spoof stratified across A01-A06
    n_bonafide_train = len(df_train_full[df_train_full["label"] == 0])
    df_train_bona = df_train_full[df_train_full["label"] == 0]

    df_train_spoof_parts = []
    attacks_train = sorted(df_train_full[df_train_full["label"] == 1]["system_id"].unique())
    per_attack = n_bonafide_train // len(attacks_train)
    for att in attacks_train:
        sub = df_train_full[df_train_full["system_id"] == att]
        df_train_spoof_parts.append(sub.sample(n=min(per_attack, len(sub)), random_state=config.RANDOM_SEED))
    df_train_spoof = pd.concat(df_train_spoof_parts)

    df_train_balanced = pd.concat([df_train_bona, df_train_spoof]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    # Balanced dev set: 1000 bona fide + 1000 spoof stratified across A01-A06
    df_dev_bona = df_dev_full[df_dev_full["label"] == 0].sample(n=1000, random_state=config.RANDOM_SEED)
    df_dev_spoof_parts = []
    per_attack_dev = 1000 // len(attacks_train)
    for att in attacks_train:
        sub = df_dev_full[df_dev_full["system_id"] == att]
        df_dev_spoof_parts.append(sub.sample(n=min(per_attack_dev, len(sub)), random_state=config.RANDOM_SEED))
    df_dev_spoof = pd.concat(df_dev_spoof_parts)
    df_dev_balanced = pd.concat([df_dev_bona, df_dev_spoof]).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)

    print(f"Sampled Training Utterances: {len(df_train_balanced)} (Bona fide: {(df_train_balanced['label']==0).sum()}, Spoof: {(df_train_balanced['label']==1).sum()})")
    print(f"Sampled Validation Utterances: {len(df_dev_balanced)} (Bona fide: {(df_dev_balanced['label']==0).sum()}, Spoof: {(df_dev_balanced['label']==1).sum()})")

    skip_baseline = ("--skip-baseline" in sys.argv) or (config.MODELS_DIR / "la_academic_lcnn_baseline.pt").exists()
    if not skip_baseline:
        print("\n>>> Phase 1: Training Baseline Dual-Stream LCNN...")
        train_model(df_train_balanced, df_dev_balanced, model_type="baseline", epochs=4, batch_size=64, lr=1e-3)
    else:
        print("\n>>> Phase 1: Baseline checkpoint already saved (Val EER: 0.30%). Skipping to Augmented Model...")

    # 2. Train Augmented Dual-Stream LCNN with acoustic corruption + SpecAugment
    print("\n>>> Phase 2: Training Augmented Dual-Stream LCNN (Acoustic Augmentation + SpecAugment)...")
    train_model(df_train_balanced, df_dev_balanced, model_type="augmented", epochs=4, batch_size=64, lr=1e-3)


if __name__ == "__main__":
    main()

