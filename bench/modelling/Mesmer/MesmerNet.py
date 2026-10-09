import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from deepcell.model_zoo.panopticnet import PanopticNet
from deepcell import losses as dc_losses


IMG_SIZE = 256  # PanopticNet requires a static, square, power-of-2 input shape --
                # matches this repo's standard GenericDataset crop size.
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 2)  # (nuclear, membrane) -- Mesmer's native 2ch contract


def build_fresh(input_shape=INPUT_SHAPE, backbone="resnet50"):
    """Untrained PanopticNet, whole-cell compartment only: head 0 = inner-distance
    (1ch regression), head 1 = fgbg (2ch classification). Mesmer's nuclear branch
    (2 more heads) is dropped -- this repo has no nuclear instance masks to supervise
    it with. `use_imagenet=False`: backbone weights are also randomly initialized, no
    ImageNet download."""
    return PanopticNet(
        backbone=backbone,
        input_shape=input_shape,
        num_semantic_classes=[1, 2],
        location=True,
        use_imagenet=False,
    )


def to_mesmer_batch(img, targets):
    """img: (N,2,H,W) float32 torch tensor (nuclear, membrane) -- already sliced to
    2ch by Generic_Dataset.target_Mesmer (model_type="Mesmer"). targets: dict of
    already-batched tensors from that same method --
    {"inner_distance": (N,1,H,W), "fgbg": (N,2,H,W)}. Just a layout conversion to
    TF's channel-last numpy, matching build_fresh()'s 2 outputs -- the actual target
    construction (deepcell's own _transform_masks) lives in Generic_Dataset now, the
    same place every other model's target construction lives."""
    img_np = img.numpy().transpose(0, 2, 3, 1).astype(np.float32)  # (N,H,W,2)
    y_inner = targets["inner_distance"].numpy().transpose(0, 2, 3, 1).astype(np.float32)
    y_fgbg = targets["fgbg"].numpy().transpose(0, 2, 3, 1).astype(np.float32)
    return img_np, [y_inner, y_fgbg]


def save_model(model, filepath):
    """Weights only (`.h5`), matching every PyTorch model's `save_model`
    convention here (state_dict, not full pickle) -- avoids Keras having to
    (de)serialize this file's custom loss functions on load, which a full
    `model.save()` would need `custom_objects=` for."""
    model.save_weights(filepath)


def load_model(filepath, input_shape=INPUT_SHAPE, backbone="resnet50"):
    model = build_fresh(input_shape=input_shape, backbone=backbone)
    model.load_weights(filepath)
    return model


def train_model(model, train_loader, val_loader, model_name, save_path,
                 epochs=200, lr=1e-4, early_stopping_patience=25, save_every=20):
    """Generic training loop shared by Stage_1/Stage_2/Experiments: compiles
    smooth_l1 (distance head) + weighted_categorical_crossentropy (fgbg head), then a
    plain train_on_batch/test_on_batch loop over full passes of
    train_loader/val_loader.

    Checkpointing mirrors InstanSegNet.train_model: best-val-loss checkpoint
    (`{model_name}_BEST.h5`) plus periodic `{model_name}_{epoch}.h5` saves, with
    early stopping on val loss. No hotstart phase (that's an InstanSeg-loss-specific
    two-phase schedule, not applicable here).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=[dc_losses.smooth_l1, dc_losses.weighted_categorical_crossentropy],
        loss_weights=[1.0, 1.0],
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    best_val_loss = float("inf")
    early_stop_counter = 0
    for epoch in range(epochs):
        train_losses = [model.train_on_batch(*to_mesmer_batch(img, targets))
                         for img, targets in train_loader]
        train_loss = float(np.mean([l[0] for l in train_losses]))
        msg = f"epoch {epoch}: train_loss={train_loss:.4f}"

        val_loss = None
        if val_loader is not None:
            val_losses = [model.test_on_batch(*to_mesmer_batch(img, targets))
                          for img, targets in val_loader]
            val_loss = float(np.mean([l[0] for l in val_losses]))
            msg += f" val_loss={val_loss:.4f}"
        print(msg)

        if val_loss is not None and save_path:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                save_model(model, os.path.join(save_path, f"{model_name}_BEST.h5"))
            elif early_stopping_patience > 0:
                early_stop_counter += 1
                if early_stop_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}, best model {early_stopping_patience} epochs ago")
                    break

        if save_path and ((epoch + 1) == epochs or (epoch + 1) % save_every == 0):
            save_model(model, os.path.join(save_path, f"{model_name}_{epoch + 1}.h5"))

    if save_path:
        best_path = os.path.join(save_path, f"{model_name}_BEST.h5")
        if os.path.exists(best_path):
            model.load_weights(best_path)
            print(f"Reloaded best checkpoint: {best_path}")

    return model
