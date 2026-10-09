from pathlib import Path
import sys
import csv
import io
import numpy as np

from scripts.utils import ensure_parent, read_image


def _get_metrics(gt: np.ndarray, pred: np.ndarray, iou_thr: float) -> dict:
    bench_dir = Path(__file__).resolve().parents[1] / "bench"
    if str(bench_dir) not in sys.path:
        sys.path.append(str(bench_dir))
    from validation.object_level import ObjectLevelMetrics

    metrics = ObjectLevelMetrics(gt, pred, iou_thr=iou_thr)
    return {
        "PQ": float(metrics.pq),
        "F1": float(metrics.f1),
        "mIoU": float(metrics.mean_mask_iou()),
        "DICE": float(metrics.mean_dice()),
    }

def _write_csv(path: Path, row: dict) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    path.write_text(buffer.getvalue())


def main() -> None:
    output_path = Path(snakemake.output[0])
    ensure_parent(output_path)

    gt_path = getattr(snakemake.params, "gt", None)
    if not snakemake.params.enabled or gt_path in (None, "None", ""):
        _write_csv(output_path, {"skipped": True, "reason": "no_reference_or_disabled"})
        return

    pred = read_image(snakemake.input.mask)
    gt = read_image(gt_path)
    iou_thr = float(getattr(snakemake.params, "iou_threshold", 0.5))

    all_metrics = _get_metrics(gt, pred, iou_thr=iou_thr)
    requested = [m.lower() for m in getattr(snakemake.params, "metrics", [])]
    if requested:
        aliases = {
            "iou": "miou",
            "miou": "miou",
            "dice": "dice",
            "mdice": "dice",
            "pq": "pq",
            "f1": "f1",
        }
        requested_keys = {aliases.get(name, name) for name in requested}
        metrics = {k: v for k, v in all_metrics.items() if k.lower() in requested_keys}
    else:
        metrics = all_metrics

    row = {
        "PQ": metrics.get("PQ"),
        "F1": metrics.get("F1"),
        "mIoU": metrics.get("mIoU"),
        "DICE": metrics.get("DICE"),
        "gt": str(gt_path),
        "mask": str(snakemake.input.mask),
        "iou_threshold": iou_thr,
    }
    _write_csv(output_path, row)

if __name__ == "__main__":
    main()
