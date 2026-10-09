"""Cellpose multiplexed_input benchmark for Vectra.

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

from inferencing.Cellpose_inference import infer_stack
from modelling.Cellpose.myCellpose import myCellpose
from training.common.pipeline import run_kfold_pipeline, evaluate_model
from training.common import channels as ch

DATASET = "Vectra"
METHOD = "multiplexed_input"
METHOD_TAG = "Input"
ARMS = ("selected", "all")

BASE_PATH = str(data_path("Multiplex/Vectra/"))
DATASET_NAMES = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08",
                 "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
MODEL_TYPE = "CPose2ch"

PRETRAINED_PATH = str(BENCH_DIR / "training/Cellpose/Stage_2/CPose2ch_STAGE2_full_BEST.pt")
SAVE_PATH = str(BENCH_DIR / "training/Cellpose/Experiments/Vectra/runs/")
CSV_PATH = str(BENCH_DIR / "training/Cellpose/Experiments/Vectra/scores_n/")

RANDOM_STATE = 42
BATCH_SIZE = 8
LEARNING_RATE = 1e-3

def rebuild_first_layer(co, n_ch):
    """Swap the first residual-down block's input convs (and their BatchNorms) to accept
    `n_ch` channels instead of the pretrained checkpoint's 2."""
    co.net.downsample.down.res_down_0.conv.conv_0[0] = nn.BatchNorm2d(
        n_ch, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
    co.net.downsample.down.res_down_0.conv.conv_0[2] = nn.Conv2d(
        n_ch, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    co.net.downsample.down.res_down_0.proj[0] = nn.BatchNorm2d(
        n_ch, eps=1e-05, momentum=0.05, affine=True, track_running_stats=True)
    co.net.downsample.down.res_down_0.proj[1] = nn.Conv2d(
        n_ch, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    return co


def train_model(co, train_loader, val_loader, model_name, save_path, lr,
                epochs_frozen=100, epochs_full=200):
    """Train the fresh input-facing layers alone, then unfreeze the whole net."""
    for param in co.net.parameters():
        param.requires_grad = False
    for param in co.net.downsample.down.res_down_0.conv.conv_0.parameters():
        param.requires_grad = True
    for param in co.net.downsample.down.res_down_0.proj.parameters():
        param.requires_grad = True

    co._train_net(train_loader, test_loader=val_loader,
                  early_stopping_patience=25, learning_rate=lr, n_epochs=epochs_frozen,
                  model_name=model_name + "_frozen", save_path=save_path, save_every=5)

    for param in co.net.parameters():
        param.requires_grad = True

    co._train_net(train_loader, test_loader=val_loader,
                  early_stopping_patience=25, learning_rate=lr, n_epochs=epochs_full,
                  model_name=model_name + "_full", save_path=save_path, save_every=5)
    return co



def build_model(n_ch):
    """Build the model for `n_ch` input channels (arm-dependent)."""
    co = myCellpose(nchan=2, gpu=True, pretrained_model=PRETRAINED_PATH)
    co = rebuild_first_layer(co, n_ch)
    co.net.to("cuda")
    return co



def run_segmentation_pipeline(arm, fold, base_path=BASE_PATH, dataset_names=DATASET_NAMES,
                              save_path=SAVE_PATH, csv_path=CSV_PATH,
                              batch_size=BATCH_SIZE, epochs_frozen=100, epochs_full=200, lr=None,
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
                           save_path=save_path, lr=lr or LEARNING_RATE,
                           epochs_frozen=epochs_frozen, epochs_full=epochs_full)

    trained_model, test_dataset = run_kfold_pipeline(
        img_paths, label_paths, funcs, dapi_idxs, MODEL_TYPE, n_ch,
        fold, batch_size, train_fn, n_splits_total=n_splits_total, random_state=random_state,
    )

    trained_model.net.to("cpu")
    evaluate_model(trained_model, test_dataset, csv_path, model_name, infer_fn=infer_stack)
    print(f"[{model_name}] done ({n_ch} input channels)")
    return trained_model


def run_pipeline(arm="selected", fold=0, **kwargs):
    print(f"Running Cellpose multiplexed_input on {DATASET} | arm={arm} | fold={fold}")
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
