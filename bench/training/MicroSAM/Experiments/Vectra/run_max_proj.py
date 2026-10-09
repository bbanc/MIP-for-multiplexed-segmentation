"""MicroSAM zero-shot max_proj benchmark for Vectra.

Channel arms come from `training/common/channels.py`:
  "selected" - up to 7 curated markers (DAPI first)
  "all"      - every marker in the raw stack, zero-inflated to the dataset's largest panel

Run one arm/fold directly, or `run_all()` for both arms x 5 folds.

2026-08-24: added arm support. Previously this always built its MaxProj input
via `img_stack_max` maxing over every raw channel with no curation -- already
equivalent to today's "all" arm (zero-padding before a max doesn't change the
max, so `channels.py`'s zero-inflated "all" arm and the old raw max agree
exactly), just untagged. Existing untagged `{model_type}_ZeroShot_Vectra_MaxProj_fold{N}`
runs are kept as-is and read as the "all" arm (see `bench/validation/channel_arm_bars.py`);
only "selected" needed a real new run.
"""
import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from import_util import data_path
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from inferencing.MicroSAM_inference import infer_img_zero_shot
from modelling.MicroSAM.MicroSAMNet import build_pretrained
from training.common.pipeline import build_kfold_datasets, evaluate_model
from training.common import channels as ch

DATASET = "Vectra"
METHOD = "max_proj"
METHOD_TAG = "MaxProj"
ARMS = ("selected", "all")

BASE_PATH = str(data_path("Multiplex/Vectra/"))
DATASET_NAMES = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16']
MODEL_TYPE = "MicroSAM"

CSV_PATH = str(BENCH_DIR / "training/MicroSAM/Experiments/Vectra/scores_n/")

RANDOM_STATE = 42


def run_segmentation_pipeline(arm, fold, base_path=BASE_PATH, dataset_names=DATASET_NAMES,
                              csv_path=CSV_PATH, n_splits_total=5, random_state=RANDOM_STATE):
    os.makedirs(csv_path, exist_ok=True)

    img_paths, label_paths, funcs, dapi_idxs = ch.build_paths(
        DATASET, dataset_names, base_path, arm, METHOD
    )
    n_ch = ch.n_input_channels(DATASET, arm, METHOD)

    # Same KFold split/seed as the trained models -- test-set membership matches theirs, so
    # per-fold results stay comparable via bench/validation/multi_model.py.
    _, test_dataset = build_kfold_datasets(
        img_paths, label_paths, funcs, dapi_idxs, MODEL_TYPE, n_ch,
        fold, n_splits_total=n_splits_total, random_state=random_state,
    )

    model_name = f"{MODEL_TYPE}_ZeroShot_{DATASET}_MaxProj_{arm}_fold{fold}"
    model = build_pretrained()
    evaluate_model(model, test_dataset, csv_path, model_name, infer_fn=infer_img_zero_shot)
    print(f"[{model_name}] zero-shot pipeline done ({n_ch} input channels)")
    return model


def run_pipeline(arm="selected", fold=0, **kwargs):
    print(f"Running {MODEL_TYPE} zero-shot max_proj on {DATASET} | arm={arm} | fold={fold}")
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
