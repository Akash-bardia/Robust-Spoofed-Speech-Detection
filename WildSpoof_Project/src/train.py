import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, dataset, models, evaluate, utils

def train_cnn(X_train_mel, y_train, X_val_mel, y_val, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE, epochs=config.EPOCHS):
    utils.set_seed(config.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CNN model on device: {device}")

    train_ds = dataset.SpoofDataset(X_train_mel, y_train)
    val_ds = dataset.SpoofDataset(X_val_mel, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = models.SpoofCNN().to(device)

    # Calculate class weights if imbalanced
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_model_path = config.MODELS_DIR / "best_cnn_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= len(train_ds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_logits_list, val_targets_list = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_logits_list.append(logits.cpu().numpy())
                val_targets_list.append(batch_y.cpu().numpy())

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        val_logits = np.concatenate(val_logits_list)
        val_scores = 1.0 / (1.0 + np.exp(-val_logits))  # Sigmoid probabilities
        val_preds = (val_scores >= 0.5).astype(int)
        val_targets = np.concatenate(val_targets_list)
        val_acc = (val_preds == val_targets).mean()

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path}")

    # Load best checkpoint for inference
    model.load_state_dict(torch.load(best_model_path))
    return model, device

def predict_cnn(model, device, X_mel, batch_size=config.BATCH_SIZE):
    model.eval()
    ds = dataset.SpoofDataset(X_mel, np.zeros(len(X_mel)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    logits_list = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            logits_list.append(logits.cpu().numpy())
            
    logits = np.concatenate(logits_list)
    scores = 1.0 / (1.0 + np.exp(-logits))
    preds = (scores >= 0.5).astype(int)
    return preds, scores
