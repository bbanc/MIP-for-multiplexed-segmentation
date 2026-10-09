import os

import torch
from torch import nn
from cellpose import models
from cellpose.vit import CPSAM


class CPSAMModel:
    def __init__(self, device=None, dtype=torch.float32):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.net = CPSAM(dtype=dtype)
        self.net.to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.net.to(self.device)
        return self

    def save_model(self, filepath):
        torch.save(self.net.state_dict(), filepath)

    def load_model(self, filepath):
        self.net.load_state_dict(torch.load(filepath, map_location=self.device, weights_only=True))

    def _compute_masks(self, *args, **kwargs):
        return models.CellposeModel._compute_masks(self, *args, **kwargs)


def build_fresh(device=None):
    return CPSAMModel(device=device)


def build_pretrained(device=None, pretrained_model="cpsam_v2"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model=pretrained_model, device=device)


def loss_fn(lbl, y, device):
    criterion = nn.MSELoss(reduction="mean")
    criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
    veci = (5. * lbl[:, 1:]).to(device)
    binary_lbl = (lbl[:, 0] > .5).float().to(device)
    loss = criterion(y[:, :2], veci) / 2.
    loss = loss + criterion2(y[:, 2], binary_lbl)
    return loss


def train_model(model, train_loader, val_loader, model_name, save_path,
                 n_epochs=100, learning_rate=1e-5, weight_decay=0.1,
                 early_stopping_patience=10, save_every=20):
    device = model.device
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(n_epochs):
        model.net.train()
        train_losses = []
        for imgi, lbl in train_loader:
            x = imgi.float().to(device)
            optimizer.zero_grad()
            y = model.net(x)[0]
            loss = loss_fn(lbl, y, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / max(len(train_losses), 1)
        msg = f"epoch {epoch}: train_loss={train_loss:.4f}"

        val_loss = None
        if val_loader is not None:
            model.net.eval()
            val_losses = []
            with torch.no_grad():
                for imgi, lbl in val_loader:
                    x = imgi.float().to(device)
                    y = model.net(x)[0]
                    val_losses.append(loss_fn(lbl, y, device).item())
            val_loss = sum(val_losses) / max(len(val_losses), 1)
            msg += f" val_loss={val_loss:.4f}"
        print(msg)

        if val_loss is not None and save_path:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                model.save_model(os.path.join(save_path, f"{model_name}_BEST.pt"))
            elif early_stopping_patience > 0:
                early_stop_counter += 1
                if early_stop_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}, best model {early_stopping_patience} epochs ago")
                    break

        if save_path and ((epoch + 1) == n_epochs or (epoch + 1) % save_every == 0):
            model.save_model(os.path.join(save_path, f"{model_name}_{epoch + 1}.pt"))

    return model
