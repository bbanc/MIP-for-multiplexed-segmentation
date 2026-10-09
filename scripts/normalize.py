from pathlib import Path
import numpy as np

from scripts.utils import list_images, read_image, write_image


def _normalize_intensity(arr: np.ndarray, method: str, params: dict) -> np.ndarray:
    method = (method or "").lower()
    eps = 1e-8

    if method == "minmax":
        mins = arr.min(axis=(0, 1))
        maxs = arr.max(axis=(0, 1))
        scale = np.maximum(maxs - mins, eps)
        arr = (arr - mins) / scale
        arr = np.clip(arr, 0.0, 1.0)

    elif method == "percentile":
        low, high = params.get("percentiles", [1.0, 99.0])
        lows = np.percentile(arr, low, axis=(0, 1))
        highs = np.percentile(arr, high, axis=(0, 1))
        arr = np.clip(arr, lows, highs)
        scale = np.maximum(highs - lows, eps)
        arr = (arr - lows) / scale
        arr = np.clip(arr, 0.0, 1.0)

    else:
        raise ValueError(f"Unsupported intensity normalization method: {method}")

    return arr


def _get_channel_paths(images_folder: Path, marker_names: list[str], dapi_name: str) -> list[Path]:
    """Select only DAPI and marker channels that will be used for max projection.
    Skip if no DAPI found.
    """

    all_paths = list_images(images_folder)
    selected: list[Path] = []
    dapi_path = None
    if dapi_name:
        dapi_path = images_folder / dapi_name
        if dapi_path.exists():
            selected.append(dapi_path)
    for p in all_paths:
        if (dapi_path is not None) & (p == dapi_path):
            continue
        if any(name.lower() in p.name.lower() for name in marker_names):
            selected.append(p)
    return selected


def main() -> None:
    output_path = Path(snakemake.output[0])
    output_path.mkdir(parents=True, exist_ok=False)

    artifact_mask_path = snakemake.params.artifact_removal.get("mask", None)
    intensity_params = snakemake.params.intensity

    img_folder = Path(snakemake.input[0])
    marker_sel = snakemake.config.get("max_projection", {}).get("marker_selection", [])

    dapi_param = snakemake.params.get("dapi_channel", None)
    dapi_name = Path(str(dapi_param)).name if dapi_param is not None else None # Base name

    img_paths = _get_channel_paths(img_folder, marker_sel, dapi_name)

    mask = read_image(artifact_mask_path) if artifact_mask_path else None
    for img_p in img_paths:
        img = read_image(img_p)
        if mask is not None:
            if mask.shape[:2] != img.shape[:2]:
                raise ValueError(
                    f"Artifact mask shape {mask.shape[:2]} does not match image shape {img.shape[:2]}"
                )
            img[mask != 0] = 0
        img = _normalize_intensity(img, intensity_params.get("method", "minmax"), intensity_params)
        file_name = img_p.stem + ".tif"
        channel_path = output_path / file_name
        write_image(channel_path, (img * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
