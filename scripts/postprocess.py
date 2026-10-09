from pathlib import Path

import numpy as np
import skimage
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import expand_labels

from scripts.utils import read_image, write_image


def fill_small_holes(labels: np.ndarray, max_size: int) -> np.ndarray:
    """Give background pockets of at most `max_size` px that are enclosed by cells to the nearest cell."""
    pockets = remove_small_holes(labels > 0, max_size=max_size) & (labels == 0)
    nearest = expand_labels(labels, distance=int(np.ceil(np.sqrt(max_size))))
    filled = labels.copy()
    filled[pockets] = nearest[pockets]
    return filled


def main() -> None:
    output_path = Path(snakemake.output[0])
    param = snakemake.params
    remove_small = int(param.get("remove_small", 0) or 0)
    fill_holes = int(param.get("fill_holes", 0) or 0)
    labels = read_image(snakemake.input[0])

    if remove_small > 0:
        labels = remove_small_objects(labels, max_size=remove_small)
    if fill_holes > 0:
        labels = fill_small_holes(labels, fill_holes)
    out_mask = skimage.measure.label(labels).astype(np.uint16)
    write_image(output_path, out_mask)


if __name__ == "__main__":
    main()
