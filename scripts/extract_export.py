from pathlib import Path
from typing import List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import skimage

from scripts.utils import ensure_parent, list_images, read_image


def _label_ids(labels: np.ndarray) -> np.ndarray:
    label_ids = np.unique(labels)
    if label_ids.size:
        label_ids = label_ids[label_ids != 0]
    return label_ids.astype(np.int32, copy=False)


def _compute_intensity_features(
    labels: np.ndarray,
    label_ids: np.ndarray,
    img_paths: List[Path],
) -> Tuple[np.ndarray, List[str]]:
    if label_ids.size == 0 or not img_paths:
        return np.empty((label_ids.size, 0), dtype=np.float32), []

    means = np.empty((label_ids.size, len(img_paths)), dtype=np.float32)
    col_names: List[str] = []

    for i, f in enumerate(sorted(img_paths)):
        img = read_image(f)
        means[:, i] = ndimage.mean(img, labels=labels, index=label_ids).astype(np.float32, copy=False)
        col_names.append(Path(f).stem)

    return means, col_names


def _compute_centroids_and_shapes(
    labels: np.ndarray,
    label_ids: np.ndarray,
    export_shapes: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    n_labels = label_ids.size
    x = np.full(n_labels, np.nan, dtype=np.float32)
    y = np.full(n_labels, np.nan, dtype=np.float32)
    shape_features = np.empty((n_labels, 0), dtype=np.float32)
    shape_names: List[str] = []

    if n_labels == 0:
        return x, y, shape_features, shape_names

    properties = ["label", "centroid"]
    if export_shapes:
        properties += ["area", "perimeter", "eccentricity", "solidity"]

    props = skimage.measure.regionprops_table(labels, properties=tuple(properties))
    if not props or len(props.get("label", [])) == 0:
        return x, y, shape_features, shape_names

    labels_present = np.asarray(props["label"], dtype=label_ids.dtype)
    cent_y = np.asarray(props["centroid-0"], dtype=np.float32)
    cent_x = np.asarray(props["centroid-1"], dtype=np.float32)
    if labels_present.shape[0] == n_labels and np.array_equal(labels_present, label_ids):
        y = cent_y
        x = cent_x
        if export_shapes:
            area = np.asarray(props["area"], dtype=np.float32)
            perimeter = np.asarray(props["perimeter"], dtype=np.float32)
            eccentricity = np.asarray(props["eccentricity"], dtype=np.float32)
            solidity = np.asarray(props["solidity"], dtype=np.float32)
    else:
        if export_shapes:
            area = np.full(n_labels, np.nan, dtype=np.float32)
            perimeter = np.full(n_labels, np.nan, dtype=np.float32)
            eccentricity = np.full(n_labels, np.nan, dtype=np.float32)
            solidity = np.full(n_labels, np.nan, dtype=np.float32)
        idx_map = {label: i for i, label in enumerate(label_ids)}
        for j, label in enumerate(labels_present):
            idx = idx_map.get(int(label))
            if idx is None:
                continue
            y[idx] = cent_y[j]
            x[idx] = cent_x[j]
            if export_shapes:
                area[idx] = props["area"][j]
                perimeter[idx] = props["perimeter"][j]
                eccentricity[idx] = props["eccentricity"][j]
                solidity[idx] = props["solidity"][j]

    if export_shapes:
        circularity = np.full(n_labels, np.nan, dtype=np.float32)
        valid = perimeter > 0
        circularity[valid] = 4.0 * np.pi * area[valid] / (perimeter[valid] ** 2)
        shape_features = np.stack(
            [area, perimeter, circularity, eccentricity, solidity],
            axis=1,
        )
        shape_names = ["area", "perimeter", "circularity", "eccentricity", "solidity"]

    return y, x, shape_features, shape_names


def main() -> None:
    out_features = Path(snakemake.output.features)
    out_h5ad = Path(snakemake.output.h5ad)

    mask_path = snakemake.input.get("mask")
    img_folder = snakemake.input.get("raw_image_folder")
    if mask_path is None:
        raise ValueError("Missing required input: mask")

    mask = read_image(mask_path)
    if not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int32)

    label_ids = _label_ids(mask)
    export_conf = snakemake.config.get("export", {})
    export_intensity = bool(export_conf.get("intensity", True))
    export_shapes = bool(export_conf.get("shapes", False))

    y, x, shape_features, shape_names = _compute_centroids_and_shapes(mask, label_ids, export_shapes)

    feature_blocks: List[np.ndarray] = []
    var_names: List[str] = []

    if export_intensity:
        if img_folder is None:
            raise ValueError("Missing required input: raw_image_folder")
        img_files = list_images(Path(img_folder))
        intensity, intensity_names = _compute_intensity_features(mask, label_ids, img_files)
        if intensity.shape[1] > 0:
            feature_blocks.append(intensity)
            var_names.extend(intensity_names)

    if export_shapes and shape_features.shape[1] > 0:
        feature_blocks.append(shape_features)
        var_names.extend(shape_names)

    if not feature_blocks:
        raise ValueError("Nothing to export: the mask has no cells, or both export.intensity and export.shapes are off")
    X = np.concatenate(feature_blocks, axis=1)

    obs = pd.DataFrame(
        {"x": x, "y": y},
        index=pd.Index(label_ids.astype(str), name="Cell_ID"),
    )
    var = pd.DataFrame(index=pd.Index([str(name) for name in var_names], name="feature"))

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = np.column_stack([x, y]).astype(np.float32, copy=False)

    ensure_parent(out_features)
    ensure_parent(out_h5ad)
    df = adata.to_df()   # Keep it as close to anndata as possible, probably redundant
    df[["x", "y"]] = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)
    df.to_csv(str(out_features), index=True, header=True)
    adata.write_h5ad(str(out_h5ad), compression="gzip")



if __name__ == "__main__":
    main()
