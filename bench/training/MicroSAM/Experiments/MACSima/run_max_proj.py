"""MicroSAM zero-shot max_proj benchmark for MACSima.

Channel arms come from `training/common/channels.py`:
  "selected" - up to 7 curated markers (DAPI first)
  "all"      - every marker in the raw stack, zero-inflated to the dataset's largest panel

Run one arm/fold directly, or `run_all()` for both arms x 5 folds.

2026-08-24: added arm support. 2026-08-26: the old untagged
`{model_type}_ZeroShot_MACSima_MaxProj_fold{N}` runs were NOT an "all" arm --
MACSima's cached `MaxProj.png` is the curated/selected projection, so they
duplicated "selected". They have been deleted; both arms now have real tagged
runs (`..._selected_fold{N}` / `..._all_fold{N}`).
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

DATASET = "MACSima"
METHOD = "max_proj"
METHOD_TAG = "MaxProj"
ARMS = ("selected", "all")

BASE_PATH = str(data_path("Multiplex/MACSima/"))
DATASET_NAMES = ['Tonsils', 'Liver', 'TNBC', 'OvCa_1', 'OvCa_2', 'OvCa_3']
MODEL_TYPE = "MicroSAM"

CSV_PATH = str(BENCH_DIR / "training/MicroSAM/Experiments/MACSima/scores_n/")

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
