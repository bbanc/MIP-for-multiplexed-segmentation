from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import skimage

ArrayI = NDArray[np.int_]
ArrayF = NDArray[np.float32]

def get_bounding_boxes(label_img: ArrayI) -> ArrayI:
    """Return bounding boxes for all objects in ``label_img``.

    Box format: ``(y1, x1, y2, x2)``
    """
    props = skimage.measure.regionprops(label_img)
    if not props:
        return np.empty((0, 4), dtype=int)
    return np.array([p.bbox for p in props], dtype=int)


def bbox_iou_matrix(b1: ArrayI, b2: ArrayI, eps: float = 1e-10) -> ArrayF:
    """IoU matrix of shape (N_img1_objs, N_img1_objs)."""
    if b1.size == 0 or b2.size == 0:
        # At least one set is empty
        return np.zeros((b1.shape[0], b2.shape[0]), dtype=float)

    # Pair‑wise intersection
    y1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    x1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    y2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    x2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    inter = np.clip(y2 - y1, 0, None) * np.clip(x2 - x1, 0, None)

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + eps)


def mask_iou(mask1: ArrayI, mask2: ArrayI, eps: float = 1e-10) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter) / (union + eps)


def mask_dice(mask1: ArrayI, mask2: ArrayI, eps: float = 1e-10) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    return 2.0 * float(inter) / (mask1.sum() + mask2.sum() + eps)


def _intersection_and_areas(
    a: ArrayI, b: ArrayI, n_a: int, n_b: int
) -> Tuple[ArrayF, ArrayF, ArrayF]:
    """Pixel-intersection matrix + true per-object areas, in one O(H*W) pass.

    Encodes each pixel's (a_label, b_label) pair as one integer, then a single
    `bincount` histogram tallies every pair's overlap at once -- avoids one
    full-image scan per matched object (there can be hundreds per image).
    Areas are read off the *full* table (incl. its background row/column),
    not summed from the foreground-only submatrix, which would silently
    exclude pixels the other label doesn't cover and understate area.
    """
    combined = a.astype(np.int64) * (n_b + 1) + b.astype(np.int64)
    counts = np.bincount(combined.ravel(), minlength=(n_a + 1) * (n_b + 1)).reshape(n_a + 1, n_b + 1)
    area_a = counts.sum(axis=1)[1:].astype(np.float64)
    area_b = counts.sum(axis=0)[1:].astype(np.float64)
    inter = counts[1:, 1:].astype(np.float64)
    return inter, area_a, area_b


def _greedy_assignment(scores: ArrayF, thr: float) -> List[Tuple[int, int, float]]:
    """Greedy GT–pred matching by IoU with threshold ``thr``."""
    matched: List[Tuple[int, int, float]] = []
    scores = scores.copy()

    while True:
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        max_val = scores[idx]
        if max_val < thr:
            break
        gt_idx, pred_idx = idx
        matched.append((gt_idx, pred_idx, max_val))
        scores[gt_idx, :] = -1.0  
        scores[:, pred_idx] = -1.0  
    return matched


@dataclass
class ObjectLevelMetrics:
    """Compute metrics given two label images."""

    gt_label: ArrayI
    pred_label: ArrayI
    iou_thr: float = 0.5
    _id: Optional[str] = None
    
    iou_matrix: ArrayF = None 
    _matches: List[Tuple[int, int, float]] = None  

    def __post_init__(self) -> None:   
        self.gt_label = skimage.measure.label(self.gt_label, background=0)
        self.pred_label = skimage.measure.label(self.pred_label, background=0)

        gt_boxes = get_bounding_boxes(self.gt_label)
        pred_boxes = get_bounding_boxes(self.pred_label)
        
        self.iou_matrix = bbox_iou_matrix(gt_boxes, pred_boxes)
        self._matches = _greedy_assignment(self.iou_matrix, self.iou_thr)

    @property
    def tp(self) -> int:  # true positives
        return len(self._matches)

    @property
    def fp(self) -> int:  # false positives (unmatched predictions)
        return int(self.pred_label.max() - self.tp)

    @property
    def fn(self) -> int:  # false negatives (unmatched GT)
        return int(self.gt_label.max() - self.tp)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        denom = self.tp + 0.5 * (self.fp + self.fn)
        return self.tp / denom if denom else 0.0

    def _matched_iou_dice(self, eps: float = 1e-10) -> List[Tuple[float, float]]:
        """Per-matched-pair (iou, dice), computed via one intersection pass."""
        n_gt, n_pred = int(self.gt_label.max()), int(self.pred_label.max())
        inter, area_gt, area_pred = _intersection_and_areas(self.gt_label, self.pred_label, n_gt, n_pred)

        scores = []
        for gt_idx, pred_idx, _ in self._matches:
            i = inter[gt_idx, pred_idx]
            a_gt, a_pred = area_gt[gt_idx], area_pred[pred_idx]
            iou = i / (a_gt + a_pred - i + eps)
            dice = 2.0 * i / (a_gt + a_pred + eps)
            scores.append((iou, dice))
        return scores

    def mean_mask_iou(self) -> float:
        if not self._matches:
            return 0.0
        ious = [iou for iou, _ in self._matched_iou_dice()]
        return sum(ious) / len(ious)

    def mean_dice(self) -> float:
        if not self._matches:
            return 0.0
        dices = [dice for _, dice in self._matched_iou_dice()]
        return sum(dices) / len(dices)

    def matched_pairs(self) -> List[Tuple[int, int, float, float]]:
        """Per matched pair: ``(gt_idx, pred_idx, iou, dice)``.

        Exposes the same relabeled ``gt_idx``/``pred_idx`` and per-pair
        scores `mean_mask_iou`/`mean_dice` average over, without callers
        (e.g. a cross-model common-subset comparison) needing to duplicate
        the intersection/matching logic.
        """
        return [
            (gt_idx, pred_idx, iou, dice)
            for (gt_idx, pred_idx, _), (iou, dice) in zip(self._matches, self._matched_iou_dice())
        ]

    def threshold_stats(
        self,
        thresholds: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """Return a dict with ``Prec@thr``/``Rec@thr``/``F1@thr`` for each IoU threshold.

        Re-runs the same greedy bbox-IoU matching (`self.iou_matrix`, already
        computed once in `__post_init__`) at each threshold instead of the
        single fixed `self.iou_thr` -- a detection sweep, not just one cutoff.
        Previously mislabeled precision as "AP": true Average Precision needs
        a per-detection confidence score to rank by, which doesn't exist here
        (a label image has no per-object score), so this only ever computed
        raw precision at each threshold.
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.01, 0.1)

        out: Dict[str, float] = {}
        preds_total = self.pred_label.max()
        gts_total = self.gt_label.max()

        for thr in thresholds:
            matches = _greedy_assignment(self.iou_matrix, thr)
            tp = len(matches)
            fp = preds_total - tp
            fn = gts_total - tp

            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) else 0.0

            out[f"Prec@{thr:.1f}"] = prec
            out[f"Rec@{thr:.1f}"] = rec
            out[f"F1@{thr:.1f}"] = f1
        return out

    @property
    def pq(self) -> float:
        """Panoptic Quality = mIoU × F1."""
        return self.mean_mask_iou() * self.f1

    def as_dict(self) -> Dict[str, float]:
        d: Dict[str, Union[float, int, str]] = {
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "Prec": self.precision,
            "Rec": self.recall,
            "F1": self.f1,
            "mIoU": self.mean_mask_iou(),
            "mDICE": self.mean_dice(),
            "PQ": self.pq,
        }
        d["ID"] = self._id if self._id is not None else "UNSET"
        return d

    def __str__(self) -> str:  
        stats = self.as_dict()
        w = max(len(k) for k in stats)
        return "\n".join(
            f"{k:{w}} : {v:.4f}" if isinstance(v, float) else f"{k:{w}} : {v}"
            for k, v in stats.items()
        )


def evaluate(gt_label: ArrayI, pred_label: ArrayI, iou_thr: float = 0.4) -> Dict[str, float]:
    """Convenience wrapper: returns the same dict as ``.as_dict()``."""
    return ObjectLevelMetrics(gt_label, pred_label, iou_thr).as_dict()


def split_merge_counts(gt_label: ArrayI, pred_label: ArrayI, iou_thr: float = 0.1) -> Dict[str, int]:
    """Split/merge error decomposition: a GT object meaningfully overlapping (IoU >
    `iou_thr`) 2+ pred objects is split; a pred object meaningfully overlapping 2+ GT
    objects is a merge. `iou_thr` must stay well below 0.5 -- at 0.5 no object can have
    IoU > thr with more than one partner (a standard detection-matching fact), so
    split/merge is undefined at the fixed threshold used for Prec/Rec/F1 elsewhere. 0.1
    is the loosest rung already used in `threshold_stats`'s sweep, not a new parameter.
    """
    gt_label = skimage.measure.label(gt_label, background=0)
    pred_label = skimage.measure.label(pred_label, background=0)
    n_gt, n_pred = int(gt_label.max()), int(pred_label.max())
    inter, area_gt, area_pred = _intersection_and_areas(gt_label, pred_label, n_gt, n_pred)

    union = area_gt[:, None] + area_pred[None, :] - inter
    sig = inter / np.where(union == 0, 1, union) > iou_thr

    gt_hits = sig.sum(axis=1)
    pred_hits = sig.sum(axis=0)
    return {
        "n_split_gt": int((gt_hits >= 2).sum()),
        "n_gt": n_gt,
        "n_merge_pred": int((pred_hits >= 2).sum()),
        "n_pred": n_pred,
    }


def split_merge_sweep_counts(
    gt_label: ArrayI, pred_label: ArrayI, thresholds: Optional[Sequence[float]] = None
) -> Dict[str, int]:
    """`split_merge_counts` at every threshold in one pass -- the GT/pred overlap
    matrix (the only per-image-expensive part) is built once and reused, since
    split/merge rate necessarily -> 0 as threshold -> 0.5 (same reason `split_merge_
    counts` requires `iou_thr` well below 0.5) and the *shape* of that decay is more
    informative than a single-threshold count: a model whose split/merge rate is still
    high at a stricter threshold has more severe, not just more frequent, errors."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.5, 0.05), 2)

    gt_label = skimage.measure.label(gt_label, background=0)
    pred_label = skimage.measure.label(pred_label, background=0)
    n_gt, n_pred = int(gt_label.max()), int(pred_label.max())
    inter, area_gt, area_pred = _intersection_and_areas(gt_label, pred_label, n_gt, n_pred)
    union = area_gt[:, None] + area_pred[None, :] - inter
    iou = inter / np.where(union == 0, 1, union)

    out: Dict[str, int] = {"n_gt": n_gt, "n_pred": n_pred}
    for thr in thresholds:
        sig = iou > thr
        out[f"n_split_gt@{thr:.2f}"] = int((sig.sum(axis=1) >= 2).sum())
        out[f"n_merge_pred@{thr:.2f}"] = int((sig.sum(axis=0) >= 2).sum())
    return out


def aggregate_sweep_rows(
    rows: Sequence[Dict[str, float]],
    thresholds: Optional[Sequence[float]] = None,
) -> "pd.DataFrame":
    """Average already-computed per-image `threshold_stats()` dicts into one row per threshold.

    Split out from `sweep_dataset` so callers that already have an
    `ObjectLevelMetrics` per image (e.g. `evaluate_model`, which needs one
    anyway for the fixed-threshold scores) can reuse the aggregation without
    re-instantiating/re-matching every image a second time.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.01, 0.1)

    means = pd.DataFrame(rows).mean()
    return pd.DataFrame({
        "thr": np.round(list(thresholds), 1),
        "Prec": [means[f"Prec@{t:.1f}"] for t in thresholds],
        "Rec": [means[f"Rec@{t:.1f}"] for t in thresholds],
        "F1": [means[f"F1@{t:.1f}"] for t in thresholds],
    })


def sweep_dataset(
    pairs: Sequence[Tuple[ArrayI, ArrayI]],
    thresholds: Optional[Sequence[float]] = None,
) -> "pd.DataFrame":
    """Average Prec/Rec/F1 IoU-threshold sweep across a whole dataset of (gt, pred) pairs.

    A single image's `threshold_stats()` isn't informative on its own for
    comparing models/variants -- this averages it across every (gt, pred)
    pair in a dataset/fold, one row per threshold.
    """
    rows = [ObjectLevelMetrics(gt, pred).threshold_stats(thresholds) for gt, pred in pairs]
    return aggregate_sweep_rows(rows, thresholds)


def main():  
    gt_path = Path("./test1.png")
    pred_path = Path("./pred1.tif")

    gt = skimage.io.imread(gt_path).astype(np.uint16)
    pred = skimage.io.imread(pred_path).astype(np.uint16)

    metrics = ObjectLevelMetrics(gt, pred, iou_thr=0.4)
    print(metrics)


if __name__ == "__main__":  
    main()
