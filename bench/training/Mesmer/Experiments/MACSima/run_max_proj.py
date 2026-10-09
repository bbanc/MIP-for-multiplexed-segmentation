"""Mesmer max_proj benchmark for MACSima.

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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Mesmer: CPU/TF

from inferencing.Mesmer_inference import infer_img
from modelling.Mesmer.MesmerNet import load_model, train_model as run_mesmer_training
from training.common.pipeline import run_kfold_pipeline, evaluate_model
from training.common import channels as ch

DATASET = "MACSima"
METHOD = "max_proj"
METHOD_TAG = "MaxProj"
ARMS = ("selected", "all")

BASE_PATH = str(data_path("Multiplex/MACSima/"))
DATASET_NAMES = ["Tonsils", "Liver", "TNBC", "OvCa_1", "OvCa_2", "OvCa_3"]
MODEL_TYPE = "Mesmer"

PRETRAINED_PATH = str(BENCH_DIR / "training/Mesmer/Stage_2/checkpoints/Mesmer_STAGE2_BEST.h5")
SAVE_PATH = str(BENCH_DIR / "training/Mesmer/Experiments/MACSima/models/")
CSV_PATH = str(BENCH_DIR / "training/Mesmer/Experiments/MACSima/scores_n/")

RANDOM_STATE = 42
BATCH_SIZE = 4





def build_model(n_ch):
    """Build the model for `n_ch` input channels (arm-dependent)."""
    return load_model(PRETRAINED_PATH)



def run_segmentation_pipeline(arm, fold, base_path=BASE_PATH, dataset_names=DATASET_NAMES,
                              save_path=SAVE_PATH, csv_path=CSV_PATH,
                              batch_size=BATCH_SIZE, epochs=200, lr=1e-4, early_stopping_patience=25, save_every=20,
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
        return run_mesmer_training(model, train_loader, val_loader, model_name=model_name,
                                   save_path=save_path, epochs=epochs, lr=lr,
                                   early_stopping_patience=early_stopping_patience,
                                   save_every=save_every)

    trained_model, test_dataset = run_kfold_pipeline(
        img_paths, label_paths, funcs, dapi_idxs, MODEL_TYPE, n_ch,
        fold, batch_size, train_fn, n_splits_total=n_splits_total, random_state=random_state,
    )

    
    evaluate_model(trained_model, test_dataset, csv_path, model_name, infer_fn=infer_img)
    print(f"[{model_name}] done ({n_ch} input channels)")
    return trained_model


def run_pipeline(arm="selected", fold=0, **kwargs):
    print(f"Running Mesmer max_proj on {DATASET} | arm={arm} | fold={fold}")
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
