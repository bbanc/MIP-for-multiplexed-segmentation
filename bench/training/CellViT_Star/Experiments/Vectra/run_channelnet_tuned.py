"""Stardist channelnet_tuned benchmark for Vectra.

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

from inferencing.Stardist_inference import infer_stack
from modelling.Cellvit.AdaptorStarDistNet import build as build_adaptor_stardist
from modelling.common.AdaptorNetBase import train_staged
from training.common.pipeline import run_kfold_pipeline, evaluate_model
from training.common import channels as ch

DATASET = "Vectra"
METHOD = "channelnet_tuned"
METHOD_TAG = "ChannelNetTuned"
ARMS = ("selected", "all")

BASE_PATH = str(data_path("Multiplex/Vectra/"))
DATASET_NAMES = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08",
                 "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16"]
MODEL_TYPE = "Stardist"

PRETRAINED_PATH = str(BENCH_DIR / "training/CellViT_Star/Stage_2/checkpoints/Stardist_STAGE2_BEST.pt")
SAVE_PATH = str(BENCH_DIR / "training/CellViT_Star/Experiments/Vectra/saves/")
CSV_PATH = str(BENCH_DIR / "training/CellViT_Star/Experiments/Vectra/scores_n/")

RANDOM_STATE = 42
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = "cuda:0"




def build_model(n_ch):
    """Build the model for `n_ch` input channels (arm-dependent)."""
    model = build_adaptor_stardist(PRETRAINED_PATH, device=DEVICE)
    model.to(DEVICE)
    return model



def run_segmentation_pipeline(arm, fold, base_path=BASE_PATH, dataset_names=DATASET_NAMES,
                              save_path=SAVE_PATH, csv_path=CSV_PATH,
                              batch_size=BATCH_SIZE, epochs_adaptor_only=100, epochs_joint=200,
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
        return train_staged(model, train_loader,
                            epochs_adaptor_only=epochs_adaptor_only,
                            epochs_joint=epochs_joint, val_loader=val_loader,
                            save_path=save_path, model_name=model_name)

    trained_model, test_dataset = run_kfold_pipeline(
        img_paths, label_paths, funcs, dapi_idxs, MODEL_TYPE, n_ch,
        fold, batch_size, train_fn, n_splits_total=n_splits_total, random_state=random_state,
    )

    trained_model.to("cpu")
    evaluate_model(trained_model, test_dataset, csv_path, model_name, infer_fn=infer_stack)
    print(f"[{model_name}] done ({n_ch} input channels)")
    return trained_model


def run_pipeline(arm="selected", fold=0, **kwargs):
    print(f"Running Stardist channelnet_tuned on {DATASET} | arm={arm} | fold={fold}")
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
