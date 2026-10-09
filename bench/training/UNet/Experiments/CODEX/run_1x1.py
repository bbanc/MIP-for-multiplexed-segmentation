"""Dist U-Net 1x1 benchmark for CODEX.

Channel arms come from `training/common/channels.py`:
  "selected" - up to 7 curated markers (DAPI first)
  "all"      - every marker in the raw stack, zero-inflated to the dataset's largest panel

Run one arm/fold directly, or `run_all()` for both arms x 5 folds.
"""
import os
import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from import_util import data_path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import lightning as pl

from inferencing.DIST_inference import infer_stack
from modelling.UNet.DistNet import DIST_Net_Lightning
from training.common.pipeline import run_kfold_pipeline, evaluate_model
from training.common import channels as ch

DATASET = "CODEX"
METHOD = "1x1"
METHOD_TAG = "1x1"
ARMS = ("selected", "all")

BASE_PATH = str(data_path("Multiplex/CODEX/"))
DATASET_NAMES = ["LN", "Tnsl"]
MODEL_TYPE = "DIST"

PRETRAINED_PATH = str(BENCH_DIR / "training/UNet/Stage_2/DIST_STAGE2.ckpt")
SAVE_PATH = str(BENCH_DIR / "training/UNet/Experiments/CODEX/models/")
CSV_PATH = str(BENCH_DIR / "training/UNet/Experiments/CODEX/scores_n/")

RANDOM_STATE = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-3


def train_model(model, train_loader, val_loader, model_name, save_path, n_ch,
                epochs_frozen=100, epochs_full=200):
    """Train the fresh 1x1 front alone, then unfreeze and fine-tune everything."""
    model.checkpoint_dir = save_path
    model.lr = LEARNING_RATE

    model.add_1x1_front(n_ch)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.model.new_layer_1.parameters():
        param.requires_grad = True
    for param in model.model.new_layer_2.parameters():
        param.requires_grad = True

    model.checkpoint_name = f"{model_name}_frozen"
    trainer = pl.Trainer(max_epochs=epochs_frozen, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(os.path.join(save_path, f"{model_name}_frozen.ckpt"))

    for param in model.parameters():
        param.requires_grad = True

    model.checkpoint_name = f"{model_name}_full"
    trainer = pl.Trainer(max_epochs=epochs_full, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(os.path.join(save_path, f"{model_name}_full.ckpt"))
    return model



def build_model(n_ch):
    """Build the model for `n_ch` input channels (arm-dependent)."""
    model = DIST_Net_Lightning.load_from_checkpoint(
        arch="unet", encoder_name="resnet50", in_channels=3, out_classes=2,
        checkpoint_path=PRETRAINED_PATH)
    model.to("cuda")
    return model



def run_segmentation_pipeline(arm, fold, base_path=BASE_PATH, dataset_names=DATASET_NAMES,
                              save_path=SAVE_PATH, csv_path=CSV_PATH,
                              batch_size=BATCH_SIZE, epochs_frozen=100, epochs_full=200,
                              n_splits_total=5, random_state=RANDOM_STATE):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    img_paths, label_paths, funcs, dapi_idxs = ch.build_paths(
        DATASET, dataset_names, base_path, arm, METHOD
    )
    n_ch = ch.n_input_channels(DATASET, arm, METHOD)
    model_name = f"{MODEL_TYPE}_{DATASET}_{METHOD_TAG}_{arm}_fold{fold}"

    model = build_model(n_ch)

    def train_fn(train_loader, val_loader):
        return train_model(model, train_loader, val_loader, model_name=model_name,
                           save_path=save_path, n_ch=n_ch,
                           epochs_frozen=epochs_frozen, epochs_full=epochs_full)

    trained_model, test_dataset = run_kfold_pipeline(
        img_paths, label_paths, funcs, dapi_idxs, MODEL_TYPE, n_ch,
        fold, batch_size, train_fn, n_splits_total=n_splits_total, random_state=random_state,
    )

    trained_model.to("cpu")
    evaluate_model(trained_model, test_dataset, csv_path, model_name, infer_fn=infer_stack)
    print(f"[{model_name}] done ({n_ch} input channels)")
    return trained_model


def run_pipeline(arm="selected", fold=0, **kwargs):
    print(f"Running Dist U-Net 1x1 on {DATASET} | arm={arm} | fold={fold}")
    return run_segmentation_pipeline(arm, fold, **kwargs)


def fold_runner(arm, fold):
    run_pipeline(arm=arm, fold=fold)


def run_all(arms=ARMS, folds=range(5), **kwargs):
    for arm in arms:
        for fold in folds:
            run_pipeline(arm=arm, fold=fold, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, action="append",
                        help="channel arm to run; repeatable, defaults to both")
    parser.add_argument("--fold", type=int, action="append",
                        help="fold to run; repeatable, defaults to 0-4")
    args = parser.parse_args()
    run_all(arms=tuple(args.arm) if args.arm else ARMS,
            folds=args.fold if args.fold else range(5))
