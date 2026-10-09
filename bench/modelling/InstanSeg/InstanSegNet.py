import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from instanseg.utils.models.InstanSeg_UNet import InstanSeg_UNet
from instanseg.utils.loss.instanseg_loss import InstanSeg as InstanSegLoss
from instanseg.utils.AI_utils import train_epoch


class InstanSegModel(nn.Module):
    """Bundles the isolated segmentation backbone with its loss/decode method.

    InstanSeg's own `InstanSeg(nn.Module)` loss class (aliased `InstanSegLoss`)
    provides both `.forward` (training loss) and `.postprocessing` (inference
    decode) -- no separate decode wrapper needed.
    """
    def __init__(self, backbone, method):
        super().__init__()
        self.backbone = backbone
        self.method = method

    def to(self, device):
        self.backbone.to(device)
        self.method.device = device
        self.method.pixel_classifier.to(device)
        return self

    def save_model(self, filepath):
        """Only `backbone`'s state_dict is persisted -- `method.pixel_classifier`
        is the same object as `backbone.pixel_classifier` (see
        `initialize_pixel_classifier`), so restoring backbone restores both."""
        torch.save(self.backbone.state_dict(), filepath)

    def load_model(self, filepath, device=None):
        device = device or next(self.backbone.parameters()).device
        self.backbone.load_state_dict(torch.load(filepath, map_location=device))


def build_fresh(device=None, in_channels=3, n_sigma=2, cells_and_nuclei=False, dim_coords=2, dim_seeds=1, window_size=128):
    """Fresh/random-init backbone + method -- no checkpoint, trains from scratch.

    `cells_and_nuclei=False` since this repo's ground truth is a single instance
    mask per crop, not separate nuclei/cell masks.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # One decoder branch producing dim_coords(2) + n_sigma(2) + dim_seeds(1) = 5 channels --
    # single branch since cells_and_nuclei=False (no separate nuclei/cell heads needed).
    backbone = InstanSeg_UNet(in_channels=in_channels, out_channels=[[2, 2, 1]])

    method = InstanSegLoss(
        n_sigma=n_sigma, cells_and_nuclei=cells_and_nuclei, dim_coords=dim_coords,
        dim_seeds=dim_seeds, device=device, window_size=window_size,
    )
    # Builds a fresh ProbabilityNet sized to match n_sigma/dim_coords/feature_engineering,
    # and attaches it to both backbone.pixel_classifier and method.pixel_classifier.
    backbone = method.initialize_pixel_classifier(backbone)

    return InstanSegModel(backbone, method)


def _as_triplet_loader(loader):
    """train_epoch expects (image_batch, labels_batch, _); ours yields (img, label)."""
    for img, label in loader:
        yield img, label, None


def _eval_epoch(model, device, val_loader, method):
    """Mean val loss over one pass, using the same `method.forward` loss as training
    (not instanseg's own `test_epoch`, which also computes F1/plots and needs a
    postprocessing_fn/iou_threshold -- more than this checkpoint-selection loop needs)."""
    model.backbone.eval()
    losses = []
    with torch.no_grad():
        for img, label, _ in _as_triplet_loader(val_loader):
            img = img.to(device)
            label = label.to(device)
            output = model.backbone(img)
            loss = method.forward(output, label.clone()).mean()
            losses.append(loss.detach().cpu().numpy())
    model.backbone.train()
    return float(np.mean(losses)) if losses else float("nan")


def train_model(model, train_loader, val_loader, model_name, save_path,
                 epochs_full=200, hotstart_epochs=10, lr=1e-3, clip=20.0,
                 early_stopping_patience=25, save_every=20):
    """Two-phase schedule matching the paper: hotstart (binary_xloss + dice_loss)
    for `hotstart_epochs`, then main training (l1_distance + lovasz_hinge).

    `model.method.num_instance_cap` already defaults to 50, matching the paper's
    K=50 cap -- no action needed.

    Checkpointing mirrors `StarDistRN50_Rnd.run_training`: best-val-loss
    checkpoint (`{model_name}_BEST.pt`) plus periodic `{model_name}_{epoch}.pt`
    saves, with early stopping on val loss during the main phase.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=lr)
    args = SimpleNamespace(on_cluster=True, clip=clip)
    method = model.method

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    if hotstart_epochs > 0:
        print(f"Hotstart for {hotstart_epochs} epochs: binary_xloss + dice_loss")
        method.update_seed_loss("binary_xloss")
        method.update_binary_loss("dice_loss")
        for epoch in range(hotstart_epochs):
            train_loss, train_time = train_epoch(
                model.backbone, device, _as_triplet_loader(train_loader),
                method.forward, optimizer, args=args,
            )
            print(f"hotstart epoch {epoch}: train_loss={np.mean(train_loss):.4f} ({train_time:.1f}s)")

        print("Switching to main training: l1_distance + lovasz_hinge")

    # Always land on the main-phase loss before the main loop below, whether or
    # not a hotstart phase ran -- otherwise hotstart_epochs=0 (e.g. Stage 2,
    # fine-tuning an already-hotstarted checkpoint) would silently run on
    # InstanSegLoss's constructor defaults (binary_xloss + lovasz_hinge), a
    # combination never otherwise used here.
    method.update_seed_loss("l1_distance")
    method.update_binary_loss("lovasz_hinge")

    best_val_loss = float("inf")
    early_stop_counter = 0
    for epoch in range(epochs_full):
        train_loss, train_time = train_epoch(
            model.backbone, device, _as_triplet_loader(train_loader),
            method.forward, optimizer, args=args,
        )
        msg = f"epoch {epoch}: train_loss={np.mean(train_loss):.4f} ({train_time:.1f}s)"

        val_loss = _eval_epoch(model, device, val_loader, method) if val_loader is not None else None
        if val_loss is not None:
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

        if save_path and ((epoch + 1) == epochs_full or (epoch + 1) % save_every == 0):
            model.save_model(os.path.join(save_path, f"{model_name}_{epoch + 1}.pt"))

    if save_path:
        best_path = os.path.join(save_path, f"{model_name}_BEST.pt")
        if os.path.exists(best_path):
            model.load_model(best_path, device=device)
            print(f"Reloaded best checkpoint: {best_path}")

    return model
