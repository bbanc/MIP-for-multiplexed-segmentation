"""Cross-model/variant common-subset segmentation-quality comparison (reviewer 2).

`ObjectLevelMetrics`'s mIoU/mDICE/PQ only average over each entry's own
TP-matched cells (`object_level.py`). When detection differs across
models/variants, each one is silently scored over a different, self-selected
subset of cells -- easier to look good on if an entry detects fewer, easier
cells. This reconstructs the same metrics restricted to the subset of GT
cells *every* compared entry actually detected, plus a stratified breakdown
by how many entries detected each cell.

Works entirely from files `evaluate_model` (`training/common/pipeline.py`)
already wrote to disk -- `{csv_dir}/{model_name}_fold{fold}/preds/
{stem}_inferred.png` next to that experiment's own `scores.csv` -- so
comparing entries never re-runs training/inference, just re-reads cached
predictions. Requires every compared entry to have scored the *same* GT
images for a given fold: same `DATASET_NAMES`/layout helper building
`label_paths` in the same order, and the same `KFold(n_splits=5,
shuffle=True, random_state=42)` (the `build_kfold_datasets` default) used to
carve out that fold's test split.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import skimage.io

from validation.object_level import ObjectLevelMetrics

IOU_THR = 0.5

# entry name -> (csv_dir, model_name_prefix); csv_dir is that experiment's
# CSV_PATH, model_name_prefix is its `model_name` string minus the
# "_fold{fold}" suffix `evaluate_model` appends via its output dir.
Entries = Dict[str, Tuple[str, str]]


def pred_path_for(csv_dir: str, model_name: str, fold: int, label_path: str) -> str:
    """Same ``{stem}_inferred.png`` naming `evaluate_model` writes preds under."""
    base, name = os.path.split(label_path)
    name, _ = os.path.splitext(name)
    if name == "Annotation":
        name = os.path.basename(base)
    return os.path.join(csv_dir, f"{model_name}_fold{fold}", "preds", f"{name}_inferred.png")


def scores_csv_for(csv_dir: str, model_name: str, fold: int) -> str:
    return os.path.join(csv_dir, f"{model_name}_fold{fold}", "scores.csv")


@dataclass
class CellRecord:
    """One GT cell (``image_idx``, ``gt_idx``) and which entries detected it, at what quality."""

    image_idx: int
    gt_idx: int
    scores: Dict[str, Tuple[float, float]]  # entry name -> (iou, dice)

    @property
    def k(self) -> int:
        return len(self.scores)


def collect_cell_records(
    label_paths: Sequence[str],
    pred_paths: Dict[str, Sequence[str]],
    iou_thr: float = IOU_THR,
) -> List[CellRecord]:
    """One `CellRecord` per GT cell that at least one entry detected.

    `ObjectLevelMetrics` relabels `gt` via `skimage.measure.label` fresh for
    every entry -- a pure function of the mask, so every entry reproduces the
    exact same `gt_idx` numbering for a given image without needing to share
    state, which is what keeps entries aligned per cell below.
    """
    entry_names = list(pred_paths)
    records: List[CellRecord] = []

    for img_idx, gt_path in enumerate(label_paths):
        gt = skimage.io.imread(gt_path)
        per_cell: Dict[int, Dict[str, Tuple[float, float]]] = {}

        for name in entry_names:
            pred = skimage.io.imread(pred_paths[name][img_idx])
            olm = ObjectLevelMetrics(gt, pred, iou_thr=iou_thr)
            for gt_idx, _pred_idx, iou, dice in olm.matched_pairs():
                per_cell.setdefault(gt_idx, {})[name] = (iou, dice)

        for gt_idx, scores in per_cell.items():
            records.append(CellRecord(image_idx=img_idx, gt_idx=gt_idx, scores=scores))

    return records


def own_scores(csv_dir: str, model_name: str, fold: int) -> Dict[str, float]:
    """Each entry's own already-computed mIoU/mDICE, averaged over its own detections."""
    df = pd.read_csv(scores_csv_for(csv_dir, model_name, fold))
    return {"mIoU_own": df["mIoU"].mean(), "mDICE_own": df["mDICE"].mean()}


def common_subset_table(records: Sequence[CellRecord], entries: Entries, fold: int) -> pd.DataFrame:
    """Per-entry mIoU/mDICE over exactly the cells *every* entry detected.

    `PQ` is intentionally excluded -- it already conflates detection (F1)
    with segmentation quality, which this table is trying to separate out.
    """
    entry_names = list(entries)
    n_entries = len(entry_names)
    common = [r for r in records if r.k == n_entries]

    rows = []
    for name in entry_names:
        csv_dir, model_name = entries[name]
        own = own_scores(csv_dir, model_name, fold)
        ious = [r.scores[name][0] for r in common]
        dices = [r.scores[name][1] for r in common]
        rows.append({
            "model": name,
            "mIoU_own": own["mIoU_own"],
            "mDICE_own": own["mDICE_own"],
            "mIoU_common": np.mean(ious) if ious else float("nan"),
            "mDICE_common": np.mean(dices) if dices else float("nan"),
            "n_common": len(common),
        })
    return pd.DataFrame(rows)


def stratified_table(records: Sequence[CellRecord], entries: Entries) -> pd.DataFrame:
    """Per (entry, k) mIoU/mDICE, where k = how many entries detected that cell.

    k == len(entries) reproduces each entry's common-subset row; k == 1 is
    the hardest stratum (only one entry ever found that cell).
    """
    entry_names = list(entries)
    rows = []
    for k in range(1, len(entry_names) + 1):
        stratum = [r for r in records if r.k == k]
        for name in entry_names:
            pairs = [r.scores[name] for r in stratum if name in r.scores]
            if not pairs:
                continue
            rows.append({
                "k": k,
                "model": name,
                "mIoU": np.mean([p[0] for p in pairs]),
                "mDICE": np.mean([p[1] for p in pairs]),
                "n": len(pairs),
            })
    return pd.DataFrame(rows)


def compare(
    entries: Entries,
    label_paths: Sequence[str],
    fold: int,
    iou_thr: float = IOU_THR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (common_subset_df, stratified_df) for `entries` on `label_paths`/`fold`."""
    pred_paths = {
        name: [pred_path_for(csv_dir, model_name, fold, lp) for lp in label_paths]
        for name, (csv_dir, model_name) in entries.items()
    }
    records = collect_cell_records(label_paths, pred_paths, iou_thr)
    return common_subset_table(records, entries, fold), stratified_table(records, entries)
