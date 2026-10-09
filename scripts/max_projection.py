from pathlib import Path
import numpy as np
from scripts.utils import ensure_parent, list_images, read_image, write_image


def max_projection_hwc(arr: np.ndarray) -> np.ndarray:
    h, w, c = arr.shape
    dapi = arr[..., 0]
    if c > 1:
        max_other = np.max(arr[..., 1:], axis=-1)
    else:
        max_other = np.zeros((h, w), dtype=arr.dtype)
    out = np.stack([dapi, max_other, np.zeros_like(dapi)], axis=-1)
    return out


def _load_select_markers(channel_paths: list[Path]) -> np.ndarray:
    if not channel_paths:
        raise ValueError("No marker images found for the requested channels")
    array = [read_image(p) for p in channel_paths]
    return np.stack(array, axis=-1)


def _get_channel_paths(images_folder: Path, marker_names: list[str], dapi_name: str | None) -> list[Path]:
    """Select only DAPI and marker channels that will be used for max projection. 
    Redundant but might help preserve sanity
    """
    all_paths = list_images(images_folder)

    selected: list[Path] = []
    dapi_path: Path | None = None
    if dapi_name is not None:
        dapi_path = images_folder / dapi_name
        if dapi_path.exists():
            selected.append(dapi_path)

    for p in all_paths:
        if dapi_path is not None and p == dapi_path:
            continue
        if any(name.lower() in p.name.lower() for name in marker_names):
            selected.append(p)

    return selected


def main() -> None:
    output_path = Path(snakemake.output[0])
    ensure_parent(output_path)

    images_folder = Path(snakemake.input[0])
    marker_sel = snakemake.params.get("marker_selection", [])

    raw_dapi = snakemake.params.get("dapi_channel", None)
    if raw_dapi is not None:
        dapi_name: str | None = Path(str(raw_dapi)).name
    else:
        dapi_name = None

    channel_paths = _get_channel_paths(images_folder, marker_sel, dapi_name)
    arr = _load_select_markers(channel_paths)
    img = max_projection_hwc(arr)
    write_image(output_path, img)


if __name__ == "__main__":
    main()
