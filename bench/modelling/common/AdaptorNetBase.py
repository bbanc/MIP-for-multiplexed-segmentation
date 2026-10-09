import os

import torch
import torch.nn as nn

from instanseg.utils.models.ChannelInvariantNet import ChannelInvariantNet


class AdaptorNetSegmenterBase(nn.Module):
    def __init__(self, segmenter: nn.Module, adaptor_kwargs: dict = None):
        super().__init__()
        self.adaptor_net = ChannelInvariantNet(**(adaptor_kwargs or {"out_channels": 3}))
        self.segmenter = segmenter

    def forward(self, x):
        x3 = self.adaptor_net(x)
        return self.segmenter(x3)

    def project(self, x):
        with torch.no_grad():
            return self.adaptor_net(x)

    def freeze_segmenter(self, freeze: bool = True):
        for p in self.segmenter.parameters():
            p.requires_grad = not freeze

    def trainable_parameters(self, stage: str):
        """stage: 'adaptor_only' or 'joint'"""
        if stage == "adaptor_only":
            self.freeze_segmenter(True)
            return self.adaptor_net.parameters()
        elif stage == "joint":
            self.freeze_segmenter(False)
            return self.parameters()
        raise ValueError(stage)

    def compute_loss(self, x, targets):
        raise NotImplementedError

    def set_loss_stage(self, stage: str):
        pass


def _move_to_device(targets, device):
    if isinstance(targets, dict):
        return {k: v.to(device) for k, v in targets.items()}
    return targets.to(device)


def _train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, targets in loader:
        x = x.to(device)
        targets = _move_to_device(targets, device)
        optimizer.zero_grad()
        loss = model.compute_loss(x, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def _eval_epoch(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, targets in loader:
            x = x.to(device)
            targets = _move_to_device(targets, device)
            loss = model.compute_loss(x, targets)
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def train_staged(model, train_loader, epochs_adaptor_only, epochs_joint,
                  lr_adaptor=1e-3, lr_segmenter=1e-3, device=None, hotstart_epochs=10,
                  val_loader=None, early_stopping_patience=25, save_path=None,
                  model_name="adaptor_model"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if hotstart_epochs > 0:
        model.set_loss_stage("hotstart")
        optimizer = torch.optim.Adam(model.trainable_parameters("adaptor_only"), lr=lr_adaptor)
        for epoch in range(hotstart_epochs):
            epoch_loss = _train_epoch(model, train_loader, optimizer, device)
            print(f"[hotstart] epoch {epoch}: loss={epoch_loss:.4f}")
        model.set_loss_stage("main")

    optimizer = torch.optim.Adam(model.trainable_parameters("adaptor_only"), lr=lr_adaptor)
    for epoch in range(epochs_adaptor_only):
        epoch_loss = _train_epoch(model, train_loader, optimizer, device)
        print(f"[adaptor_only] epoch {epoch}: loss={epoch_loss:.4f}")

    model.trainable_parameters("joint")
    optimizer = torch.optim.Adam([
        {"params": model.adaptor_net.parameters(), "lr": lr_adaptor},
        {"params": model.segmenter.parameters(), "lr": lr_segmenter},
    ])

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    best_val_loss = float("inf")
    early_stop_counter = 0
    for epoch in range(epochs_joint):
        epoch_loss = _train_epoch(model, train_loader, optimizer, device)
        msg = f"[joint] epoch {epoch}: loss={epoch_loss:.4f}"

        if val_loader is not None:
            val_loss = _eval_epoch(model, val_loader, device)
            msg += f" val_loss={val_loss:.4f}"
            print(msg)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                if save_path:
                    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_BEST.pt"))
            elif early_stopping_patience > 0:
                early_stop_counter += 1
                if early_stop_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}, best model {early_stopping_patience} epochs ago")
                    break
        else:
            print(msg)

    if save_path:
        best_ckpt = os.path.join(save_path, f"{model_name}_BEST.pt")
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt, map_location=device))

    model.eval()
    return model
