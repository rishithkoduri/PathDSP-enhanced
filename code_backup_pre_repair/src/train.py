"""
Training Engine for PathDSP-PIU and Baseline Models:
- Gradient clipping, early stopping, LR scheduling
- Metric tracking (Loss, MAE, RMSE, Pearson)
- Checkpointing and state resumption
- Configurable optimizer and loss functions
"""

import os
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super(RMSELoss, self).__init__()
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


def get_loss_fn(loss_name: str = "MSE"):
    if loss_name.upper() == "RMSE":
        return RMSELoss()
    return nn.MSELoss()


def evaluate_loader(model, dataloader, device: torch.device):
    """
    Evaluates model on dataloader, returning predictions, targets, MAE, RMSE, and Pearson correlation.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            preds = model(batch_dev).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
            if "target" in batch:
                all_targets.append(batch["target"].squeeze(-1).numpy())

    preds_arr = np.concatenate(all_preds)
    if all_targets:
        targets_arr = np.concatenate(all_targets)
        mae = float(np.mean(np.abs(preds_arr - targets_arr)))
        rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2)))
        try:
            p_val, _ = pearsonr(preds_arr, targets_arr)
            pearson_val = float(p_val) if not np.isnan(p_val) else 0.0
        except Exception:
            pearson_val = 0.0
        return {
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson_val,
            "predictions": preds_arr,
            "targets": targets_arr
        }
    return {"predictions": preds_arr}


def train_model(model: nn.Module,
                train_loader,
                val_loader,
                config: dict,
                device: torch.device,
                checkpoint_path: str = None,
                resume_from: str = None):
    """
    Full training loop with early stopping, scheduling, and checkpointing.
    """
    model.to(device)
    t_cfg = config.get("training", {})
    lr = float(t_cfg.get("learning_rate", 0.001))
    weight_decay = float(t_cfg.get("weight_decay", 1e-4))
    max_epochs = int(t_cfg.get("max_epochs", 80))
    patience = int(t_cfg.get("early_stopping_patience", 12))
    loss_fn_name = t_cfg.get("loss_fn", "MSE")
    criterion = get_loss_fn(loss_fn_name)

    opt_name = t_cfg.get("optimizer", "AdamW").upper()
    if opt_name == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "ADAMAX":
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(t_cfg.get("scheduler_factor", 0.5)),
        patience=int(t_cfg.get("scheduler_patience", 5)),
        min_lr=1e-6
    )

    start_epoch = 0
    best_val_mae = float("inf")
    best_model_state = None
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_pearson": []
    }

    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        chk = torch.load(resume_from, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        start_epoch = chk.get("epoch", 0) + 1
        best_val_mae = chk.get("best_val_mae", float("inf"))
        history = chk.get("history", history)

    logger.info(f"Starting training on {device} for up to {max_epochs} epochs (patience={patience})...")

    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_batch_losses = []

        for batch in train_loader:
            batch_dev = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()
            preds = model(batch_dev)
            targets = batch_dev["target"]
            loss = criterion(preds, targets)

            if torch.isnan(loss):
                logger.error("NaN loss encountered during training! Aborting.")
                raise FloatingPointError("Loss diverged to NaN.")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_batch_losses.append(loss.item())

        train_loss = float(np.mean(train_batch_losses))
        val_eval = evaluate_loader(model, val_loader, device)
        val_mae = val_eval["mae"]
        val_rmse = val_eval["rmse"]
        val_pearson = val_eval["pearson"]

        scheduler.step(val_mae)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_rmse ** 2)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_pearson"].append(val_pearson)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "history": history
                }, checkpoint_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter >= patience:
            logger.info(f"Epoch {epoch+1:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val PCC: {val_pearson:.4f} | Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1} (Best Val MAE: {best_val_mae:.4f})")
            break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history
