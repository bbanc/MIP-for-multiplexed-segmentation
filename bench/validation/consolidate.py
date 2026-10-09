import sys
from itertools import product
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import pandas as pd
import skimage.io

from import_util import DATA_ROOT
from validation.shape_level import ShapeLevelMetrics

CSV_PATH = BENCH_DIR / "results" / "all_runs.csv"
DATASETS = ["Vectra", "CODEX", "MACSima", "Zeiss"]
N_FOLDS = 5

# model -> (folder under training/, run-name prefix)
MODELS = {
    "Dist U-Net": ("UNet", "DIST"),
    "Stardist": ("CellViT_Star", "Stardist"),
    "Cellpose": ("Cellpose", "CPose2ch"),
    "InstanSeg": ("InstanSeg", "InstanSeg"),
    "Mesmer": ("Mesmer", "Mesmer"),
    "Cellpose-SAM": ("Cellpose_SAM", "CPoseSAM_ZeroShot"),
    "MicroSAM": ("MicroSAM", "MicroSAM_ZeroShot"),
}


def scores_dir(model, dataset):
    return BENCH_DIR / "training" / MODELS[model][0] / "Experiments" / dataset / "scores_n"


def find_run(model, dataset, modality, arm):
    """Name of the run (without _fold{k}) for this combination, or None if it was never run."""
    prefix = MODELS[model][1]
    if arm == "selected":
        suffixes = ["_selected", "_filtered", ""]  # Some runs were tagged "_filtered", or not at all
    else:
        suffixes = ["_all"]
        if prefix.endswith("ZeroShot"):
            suffixes.append("")  # No tag = Selected
    for suffix in suffixes:
        run = f"{prefix}_{dataset}_{modality}{suffix}"
        if (scores_dir(model, dataset) / f"{run}_fold0" / "scores.csv").exists():
            return run


def build(path=CSV_PATH):
    frames = []
    for model, dataset, modality, arm in product(MODELS, DATASETS, ["MaxProj", "1x1", "Input", "ChannelNetTuned"], ["selected", "all"]):
        run = find_run(model, dataset, modality, arm)
        if run is None:
            continue
        for fold in range(N_FOLDS):
            run_dir = scores_dir(model, dataset) / f"{run}_fold{fold}"
            df = pd.read_csv(run_dir / "scores.csv")
            for other in ("sweep", "shapes"):
                df = df.merge(pd.read_csv(run_dir / f"{other}.csv"), on="ID", validate="one_to_one")
            frames.append(df.assign(Model=model, Dataset=dataset, Modality=modality, Arm=arm, Fold=fold, Run=run))
    out = pd.concat(frames, ignore_index=True)
    gt = pd.DataFrame([
        {"ID": i, **{f"GT_{k}": v[0] for k, v in ShapeLevelMetrics(skimage.io.imread(i), _id=i).write_dict().items() if k != "ID"}}
        for i in out["ID"].unique()
    ])
    out = out.merge(gt, on="ID", validate="many_to_one")
    out["ID"] = out["ID"].map(lambda p: Path(p).relative_to(DATA_ROOT).as_posix())

    first =["Model", "Dataset", "Modality", "Arm", "Fold", "Run", "ID"]
    out = out[first + [c for c in out.columns if c not in first]]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out

def load():
    return pd.read_csv(CSV_PATH)

if __name__ == "__main__":
    out = build()
    print(f"wrote {len(out)} rows from {out['Run'].nunique()} runs to {CSV_PATH}")
