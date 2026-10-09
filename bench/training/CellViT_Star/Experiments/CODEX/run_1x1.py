"""Stardist 1x1 benchmark for CODEX.

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

import torch.nn as nn

from inferencing.Stardist_inference import infer_stack
from modelling.Cellvit.StarDistRN50_Rnd import StarDistRN50_Rnd
from training.common.pipeline import run_kfold_pipeline, evaluate_model
from training.common import channels as ch

DATASET = "CODEX"
METHOD = "1x1"
METHOD_TAG = "1x1"
ARMS = ("selected", "all")

BASE_PATH = str(data_path("Multiplex/CODEX/"))
DATASET_NAMES = ["LN", "Tnsl"]
MODEL_TYPE = "Stardist"

PRETRAINED_PATH = str(BENCH_DIR / "training/CellViT_Star/Stage_2/checkpoints/Stardist_STAGE2_BEST.pt")
SAVE_PATH = str(BENCH_DIR / "training/CellViT_Star/Experiments/CODEX/saves/")
CSV_PATH = str(BENCH_DIR / "training/CellViT_Star/Experiments/CODEX/scores_n/")

RANDOM_STATE = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = "cuda:0"

def add1x1front(stardist_model, in_ch):
    """Prepend two fresh 1x1 convs mapping `in_ch` -> 3 in front of the pretrained
    ResNet encoder, leaving the encoder itself untouched."""
    new_layer = nn.Conv2d(in_ch, 3, kernel_size=1)
    new_layer2 = nn.Conv2d(3, 3, kernel_size=1)
    stardist_model.encoder = nn.Sequential(new_layer, new_layer2, stardist_model.encoder)
    stardist_model.to(DEVICE)
    return stardist_model


def train_model(model, train_loader, val_loader, model_name, save_path,
                epochs_frozen=100, epochs_full=200):
    """Train the fresh input-facing layer alone, then unfreeze the whole net."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder[0].parameters():
        p.requires_grad = True
    for p in model.encoder[1].parameters():
        p.requires_grad = True

    model.run_training(train_loader, test_loader=val_loader,
                       model_name=model_name + "_frozen",
                       save_path=save_path, save_every=20, n_epochs=epochs_frozen)

    for p in model.parameters():
        p.requires_grad = True

    model.run_training(train_loader, test_loader=val_loader,
                       model_name=model_name + "_full",
                       save_path=save_path, save_every=20, n_epochs=epochs_full)
    return model



def build_model(n_ch):
    """Build the model for `n_ch` input channels (arm-dependent)."""
    model = StarDistRN50_Rnd(n_seg_cls=1, device=DEVICE, lr=LEARNING_RATE)
    model.load_model(PRETRAINED_PATH)
    model = add1x1front(model, in_ch=n_ch)
    model.to(DEVICE)
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
                           save_path=save_path, epochs_frozen=epochs_frozen,
                           epochs_full=epochs_full)

    trained_model, test_dataset = run_kfold_pipeline(
        img_paths, label_paths, funcs, dapi_idxs, MODEL_TYPE, n_ch,
        fold, batch_size, train_fn, n_splits_total=n_splits_total, random_state=random_state,
    )

    trained_model.to("cpu")
    evaluate_model(trained_model, test_dataset, csv_path, model_name, infer_fn=infer_stack)
    print(f"[{model_name}] done ({n_ch} input channels)")
    return trained_model


def run_pipeline(arm="selected", fold=0, **kwargs):
    print(f"Running Stardist 1x1 on {DATASET} | arm={arm} | fold={fold}")
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
