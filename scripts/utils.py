from pathlib import Path
from typing import Any, List
from skimage import io as skio


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png"}
    )


def read_image(p: str | Path) -> Any:
    return skio.imread(p)


def write_image(p: str | Path, arr: Any, check_contrast: bool = False) -> None:
    skio.imsave(str(p), arr, check_contrast=check_contrast)


def link_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink from src to dst, falling back to a copy.
    """
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Use a relative link where possible to make the tree more portable.
        try:
            target = src.relative_to(dst.parent)
        except ValueError:
            target = src
        dst.symlink_to(target)
    except OSError:
        img = read_image(src)
        write_image(dst, img)
