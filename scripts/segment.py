"""Shared tiling logic for segmentation models."""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from scripts.segment_models import infer, make_model, segment
from scripts.utils import ensure_parent, read_image, write_image
from skimage.measure import label as relabel


Tile = Tuple[int, int, int, int]


def _tile_slices(tile: Tile) -> Tuple[slice, slice]:
    return (slice(tile[0], tile[1]), slice(tile[2], tile[3]))


def _parse_tiling(tiling: Dict) -> Tuple[bool, int, int, int]:
    enabled = bool(tiling.get("enabled", False))
    tile_size = tiling.get("tile_size", 0)
    overlap = int(tiling.get("overlap", 0) or 0)

    if isinstance(tile_size, (list, tuple)):
        if len(tile_size) != 2:
            raise ValueError("tile_size must be an int or a (height, width) pair")
        tile_h, tile_w = int(tile_size[0]), int(tile_size[1])
    else:
        tile_h = tile_w = int(tile_size)

    return enabled, tile_h, tile_w, overlap


def _generate_grid(image_shape: Sequence[int], tile_h: int, tile_w: int, overlap: int) -> List[Tile]:
    if tile_h <= 0 or tile_w <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    height, width = int(image_shape[0]), int(image_shape[1])
    stride_y = tile_h - overlap
    stride_x = tile_w - overlap
    if stride_y <= 0 or stride_x <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    if height <= tile_h:
        y_starts = [0]
    else:
        y_starts = list(range(0, height - tile_h + 1, stride_y))
        last = height - tile_h
        if y_starts[-1] != last:
            y_starts.append(last)

    if width <= tile_w:
        x_starts = [0]
    else:
        x_starts = list(range(0, width - tile_w + 1, stride_x))
        last = width - tile_w
        if x_starts[-1] != last:
            x_starts.append(last)

    tiles: List[Tile] = []
    for y0 in y_starts:
        y1 = min(y0 + tile_h, height)
        for x0 in x_starts:
            x1 = min(x0 + tile_w, width)
            tiles.append((y0, y1, x0, x1))
    return tiles


def _owned_intervals(starts: List[int], size: int, overlap: int) -> List[Tuple[int, int]]:
    """Owned [lo, hi) range per tile start along one axis.

    Each tile owns from its start plus half the overlap up to the next tile's start plus half, so the owned
    ranges partition the axis for any starts (including a shifted last tile). The first tile owns from 0 and
    the last to the edge.
    """
    half = overlap // 2
    out = []
    for i, s in enumerate(starts):
        lo = 0 if s == 0 else s + half
        hi = size if i == len(starts) - 1 else starts[i + 1] + half
        out.append((lo, hi))
    return out


def _merge_tiles(
    tiles: Iterable[Tile],
    tile_labels: Iterable[np.ndarray],
    image_shape: Sequence[int],
    overlap: int,
) -> np.ndarray:
    """Merge per-tile labels. A cell is kept only by the tile whose owned region contains its centroid,
    so each cell is kept exactly once even where tiles overlap."""
    height, width = int(image_shape[0]), int(image_shape[1])
    tiles = list(tiles)
    ys_starts = sorted({t[0] for t in tiles})
    xs_starts = sorted({t[2] for t in tiles})
    owned_y = dict(zip(ys_starts, _owned_intervals(ys_starts, height, overlap)))
    owned_x = dict(zip(xs_starts, _owned_intervals(xs_starts, width, overlap)))

    global_labels = np.zeros((height, width), dtype=np.uint32)
    next_label = 1

    for tile, labels in zip(tiles, tile_labels):
        (ylo, yhi), (xlo, xhi) = owned_y[tile[0]], owned_x[tile[2]]
        if labels.max() == 0:
            continue

        # centroid of each tile-local label, in global coordinates
        py, px = np.nonzero(labels)
        lid = labels[py, px]
        counts = np.bincount(lid, minlength=labels.max() + 1)
        cy = np.bincount(lid, weights=py + tile[0], minlength=labels.max() + 1) / np.maximum(counts, 1)
        cx = np.bincount(lid, weights=px + tile[2], minlength=labels.max() + 1) / np.maximum(counts, 1)

        keep = np.zeros(labels.max() + 1, dtype=bool)
        present = np.flatnonzero(counts)
        present = present[present != 0]
        keep[present] = (cy[present] >= ylo) & (cy[present] < yhi) & (cx[present] >= xlo) & (cx[present] < xhi)
        keep_ids = np.flatnonzero(keep)
        if keep_ids.size == 0:
            continue

        mapping = np.zeros(labels.max() + 1, dtype=np.uint32)
        mapping[keep_ids] = np.arange(next_label, next_label + keep_ids.size, dtype=np.uint32)
        next_label += keep_ids.size

        region = global_labels[_tile_slices(tile)]
        m = keep[labels]
        region[m] = mapping[labels[m]]

    global_labels = relabel(global_labels)
    counts = np.bincount(global_labels.ravel())
    keep = counts >= 25
    if keep.size > 0:
        keep[0] = False
    keep_map = keep[global_labels]
    global_labels[~keep_map] = 0

    return global_labels


def segment_image(img: np.ndarray, model_name: str, hyper: Dict, tiling: Dict) -> np.ndarray:
    enabled, tile_h, tile_w, overlap = _parse_tiling(tiling)
    if not enabled:
        return segment(img, model_name, hyper)

    tiles = _generate_grid(img.shape, tile_h, tile_w, overlap)
    model = make_model(model_name, hyper)
    tile_labels = [
        infer(model_name, model, img[_tile_slices(tile)], hyper)
        for tile in tiles
    ]
    return _merge_tiles(tiles, tile_labels, img.shape, overlap)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    mask_to_save = mask.astype(np.uint16, copy=False)
    write_image(path, mask_to_save)


def main() -> None:
    output_path = Path(snakemake.output[0])
    ensure_parent(output_path)

    params = {
        "model": getattr(snakemake.params, "model", "cellpose"),
        "hyperparameters": getattr(snakemake.params, "hyperparameters", {}),
        "tiling": getattr(snakemake.params, "tiling", {}),
    }

    input_path = snakemake.input[0]

    img = read_image(input_path)
    model_name = str(params["model"])
    hyper_cfg = params["hyperparameters"] or {}
    model_key = model_name.lower()
    hyper = hyper_cfg.get(model_key, {}) if isinstance(hyper_cfg, dict) else {}
    tiling = params["tiling"] or {}

    masks = segment_image(img, model_name, hyper, tiling)
    _save_mask(output_path, relabel(masks))


if __name__ == "__main__":
    main()
