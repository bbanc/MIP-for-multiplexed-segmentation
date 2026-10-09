"""Example crops for the figure notebooks: one raw filtered stack, label and Max-Proj per curated crop
(`load_example`, `load_gallery_examples`), plus the 1x1-adapter model builders `misc/method_timing.py` times.

Dataset paths and stacking functions go through `training/common/channels.py` (`build_paths`, "selected" arm),
the same single source of truth every `run_*.py` script uses.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import numpy as np
import skimage.io

from data.Stacked_Multiplexed import MultiplexedDataset, max_project
from import_util import data_path

# ---------------------------------------------------------------------------
# Per-dataset example: raw filtered (7ch) stack, label and Max-Proj crop
# ---------------------------------------------------------------------------


# (example index into that dataset's img_paths, crop window (y1, y2, x1, x2))
EXAMPLE_SPEC = {
    "Vectra": (31, (0, 150, 200, 350)),
    "CODEX": (0, (200, 350, 200, 350)),
    "MACSima": (0, (250, 500, 250, 500)),
    "Zeiss": (0, (0, 300, 0, 300)),
}


def _build_vectra_paths():
    from training.UNet.Experiments.Vectra.run_multiplexed_input import BASE_PATH, DATASET_NAMES
    from training.common import channels as ch

    return ch.build_paths("Vectra", DATASET_NAMES, BASE_PATH, arm="selected", method="multiplexed_input")


def _build_codex_paths():
    from training.UNet.Experiments.CODEX.run_multiplexed_input import BASE_PATH, DATASET_NAMES
    from training.common import channels as ch

    return ch.build_paths("CODEX", DATASET_NAMES, BASE_PATH, arm="selected", method="multiplexed_input")


def _build_macsima_paths():
    from training.UNet.Experiments.MACSima.run_multiplexed_input import BASE_PATH, DATASET_NAMES
    from training.common import channels as ch

    return ch.build_paths("MACSima", DATASET_NAMES, BASE_PATH, arm="selected", method="multiplexed_input")


def _build_zeiss_paths():
    from training.UNet.Experiments.Zeiss.run_multiplexed_input import BASE_PATH, DATASET_NAMES
    from training.common import channels as ch

    return ch.build_paths("Zeiss", DATASET_NAMES, BASE_PATH, arm="selected", method="multiplexed_input")


_PATH_BUILDERS = {
    "Vectra": _build_vectra_paths,
    "CODEX": _build_codex_paths,
    "MACSima": _build_macsima_paths,
    "Zeiss": _build_zeiss_paths,
}


def _max_proj_crop(dataset: str, img_path: str, raw_crop: np.ndarray, crop) -> np.ndarray:
    """MACSima's canonical Max-Proj visualization is the cached `MaxProj.png` written
    alongside `stacked_img.tif`, not a fresh per-channel max over the "selected"-arm
    raw stack -- that arm zero-inflates to the largest real channel panel, and
    recomputing from it looks visibly grainier/more saturated than the cached
    version other MACSima figures use. Other datasets have no such cache and fall
    back to computing it directly, same as before."""
    if dataset == "MACSima":
        maxproj_path = img_path.replace("stacked_img.tif", "MaxProj.png")
        if os.path.exists(maxproj_path):
            y1, y2, x1, x2 = crop
            return skimage.io.imread(maxproj_path)[y1:y2, x1:x2, :3]
        print(f"[render_utils] missing cached MaxProj.png: {maxproj_path}")
    return max_project(raw_crop)


def _load_example_at(dataset: str, idx: int, crop, img_paths, label_paths, funcs, dapi_idxs) -> Dict[str, np.ndarray]:
    """Shared body of `load_example`/`load_gallery_examples`: raw filtered (7ch,
    DAPI-first, per-channel-normalized uint8) crop, its label, the Max-Proj
    reduction, and the precomputed frozen-ChannelNet RGB -- all for the same crop
    window of the same example image."""
    y1, y2, x1, x2 = crop
    dataset_obj = MultiplexedDataset(img_paths, label_paths, dapi_idxs, funcs)
    img, label, _ = dataset_obj.from_idx(idx, False, False, return_raw=True)

    raw_crop = img[y1:y2, x1:x2, :]
    label_crop = label[y1:y2, x1:x2]
    max_proj_crop = _max_proj_crop(dataset, str(img_paths[idx]), raw_crop, crop)

    return {
        "raw": raw_crop,
        "label": label_crop,
        "max_proj": max_proj_crop,
        "img_path": str(img_paths[idx]),
        "label_path": str(label_paths[idx]),
        "crop": (y1, y2, x1, x2),
    }


def load_example(dataset: str) -> Dict[str, np.ndarray]:
    img_paths, label_paths, funcs, dapi_idxs = _PATH_BUILDERS[dataset]()
    idx, crop = EXAMPLE_SPEC[dataset]
    return _load_example_at(dataset, idx, crop, img_paths, label_paths, funcs, dapi_idxs)


# Curated gallery examples per dataset: (label_path, crop window (y1, y2, x1, x2)).
# Originally hand-picked in the predecessor repo's ZZ_Explainer_Cellpose.ipynb;
# verified against this pipeline's data (real, currently-evaluated test images,
# in-bounds crops) before reuse here. MACSima omitted -- its entries in that
# notebook (`Selected_Merged.png`/`Annot_processed.png`) used an older data
# layout that no longer exists; `EXAMPLE_SPEC["MACSima"]` is unaffected.
GALLERY_EXAMPLES = {
    "Vectra": [
        ("Multiplex/Vectra/P02/P02-10005(53576.13955)336,251/P02-10005(53576.13955)336,251-Crop_Tif_LABELS.png", (150, 400, 150, 400)),
        ("Multiplex/Vectra/P11/P11-10003(58107.9239)0,1200/P11-10003(58107.9239)0,1200-Crop_Tif_LABELS.png", (0, 250, 0, 250)),
        ("Multiplex/Vectra/P09/P09-10001(44091.15122)1500,1608/P09-10001(44091.15122)1500,1608-Crop_Tif_LABELS.png", (0, 250, 0, 250)),
        ("Multiplex/Vectra/P02/P02-10006(55163.20460)1800,1000/P02-10006(55163.20460)1800,1000-Crop_Tif_LABELS.png", (0, 250, 0, 250)),
    ],
    "CODEX": [
        ("Multiplex/CODEX/CODEX_LN(2400,800)/CODEX_LN(2400,800)-Crop_Tif_LABELS.png", (0, 150, 0, 150)),
        ("Multiplex/CODEX/CODEX_LN(5900,2600)/CODEX_LN(5900,2600)-Crop_Tif_LABELS.png", (0, 150, 0, 150)),
        ("Multiplex/CODEX/CODEX_Tnsl(4200,5200)/Codex_Tnsl(4200,5200)-Crop_Tif_LABELS.png", (0, 150, 0, 150)),
        ("Multiplex/CODEX/CODEX_Tnsl(5600,4700)/Codex_Tnsl(5600,4700)-Crop_Tif_LABELS.png", (50, 200, 250, 400)),
    ],
    "Zeiss": [
        ("Multiplex/Zeiss/ZP-9999(0,0)39960,9739/ZP-9999(0,0)39960,9739-Crop_Tif_LABELS.png", (0, 250, 0, 250)),
        ("Multiplex/Zeiss/ZP-10001(13500,16632)12315,7165/ZP-10001(13500,16632)12315,7165-Crop_Tif_LABELS.png", (0, 250, 250, 500)),
        ("Multiplex/Zeiss/PDAC(35000,27720)6800,3050/PDAC(35000,27720)6800,3050-Crop_Tif_LABELS.png", (0, 250, 0, 250)),
        ("Multiplex/Zeiss/ZP-10002(0,0)3000,6300/ZP-10002(0,0)3000,6300-Crop_Tif_LABELS.png", (0, 250, 0, 250)),
    ],
    # 2 of ZZ_Explainer_Cellpose.ipynb's 4 MACSima picks recovered: those folders
    # still exist and are still real test images, just reprocessed under this
    # pipeline's file names (Annotation.png, not the old Annot_processed.png/
    # Selected_Merged.png). The other two (Werner_OMAP_2, Valeria_3_B1_3) are
    # genuinely gone -- those folders no longer exist on disk at all.
    "MACSima": [
        ("Multiplex/MACSima/Liver_R1_WD_RoI2_2/Annotation.png", (70,270, 0,200)),
        ("Multiplex/MACSima/OvCa_2_B2_1/Annotation.png", (256,512,256,512)),
        ("Multiplex/MACSima/Tonsils_OMAP_2/Annotation.png", (100, 300, 100, 300))
    ],
}


def load_gallery_examples(dataset: str) -> List[Dict[str, np.ndarray]]:
    """All curated examples for `dataset` from `GALLERY_EXAMPLES`."""
    img_paths, label_paths, funcs, dapi_idxs = _PATH_BUILDERS[dataset]()
    label_paths = list(label_paths)
    out = []
    for label_path, crop in GALLERY_EXAMPLES[dataset]:
        idx = label_paths.index(str(data_path(label_path)))
        out.append(_load_example_at(dataset, idx, crop, img_paths, label_paths, funcs, dapi_idxs))
    return out


# ---------------------------------------------------------------------------
# 1x1 front: the two fresh 1x1 convs each model prepends to its backbone (`run_1x1.py`).
# Different per model, so each has its own builder and its own place for the two layers.
# ---------------------------------------------------------------------------


def _build_dist_1x1(pretrained_path: str, n_ch: int):
    from modelling.UNet.DistNet import DIST_Net_Lightning
    model = DIST_Net_Lightning.load_from_checkpoint(
        arch="unet", encoder_name="resnet50", in_channels=3, out_classes=2,
        checkpoint_path=pretrained_path)
    model.add_1x1_front(n_ch)
    return model


def _build_cellpose_1x1(pretrained_path: str, n_ch: int):
    from modelling.Cellpose.myCellpose import myCellpose, add1x1front
    model = myCellpose(nchan=2, gpu=False, pretrained_model=pretrained_path)
    return add1x1front(model, n_ch)


def _build_stardist_1x1(pretrained_path: str, n_ch: int):
    """Mirrors `training/CellViT_Star/Experiments/*/run_1x1.py`'s `add1x1front`
    (a free function local to that script, not in `modelling/`) -- reimplemented
    here rather than imported, to avoid depending on one dataset's Experiments
    script as a library."""
    import torch.nn as nn
    from modelling.Cellvit.StarDistRN50_Rnd import StarDistRN50_Rnd
    model = StarDistRN50_Rnd(n_seg_cls=1, device="cpu")
    model.load_model(pretrained_path)
    new_layer = nn.Conv2d(n_ch, 3, kernel_size=1)
    new_layer2 = nn.Conv2d(3, 3, kernel_size=1)
    model.encoder = nn.Sequential(new_layer, new_layer2, model.encoder)
    return model


def _build_instanseg_1x1(pretrained_path: str, n_ch: int):
    """Mirrors `training/InstanSeg/Experiments/*/run_1x1.py`'s `add1x1front`,
    same reimplemented-not-imported rationale as `_build_stardist_1x1`."""
    import torch.nn as nn
    from modelling.InstanSeg.InstanSegNet import build_fresh
    model = build_fresh()
    model.load_model(pretrained_path)
    new_layer = nn.Conv2d(n_ch, 3, kernel_size=1)
    new_layer2 = nn.Conv2d(3, 3, kernel_size=1)
    model.backbone = nn.Sequential(new_layer, new_layer2, model.backbone)
    return model


RENDER_1X1_SPECS = {
    "Dist U-Net": dict(
        build=_build_dist_1x1,
        pretrained=str(BENCH_DIR / "training/UNet/Stage_2/DIST_STAGE2.ckpt"),
    ),
    "Stardist": dict(
        build=_build_stardist_1x1,
        pretrained=str(BENCH_DIR / "training/CellViT_Star/Stage_2/checkpoints/Stardist_STAGE2_BEST.pt"),
    ),
    "Cellpose": dict(
        build=_build_cellpose_1x1,
        pretrained=str(BENCH_DIR / "training/Cellpose/Stage_2/CPose2ch_STAGE2_full_BEST.pt"),
    ),
    "InstanSeg": dict(
        build=_build_instanseg_1x1,
        pretrained=str(BENCH_DIR / "training/InstanSeg/Stage_2/checkpoints/InstanSeg_STAGE2_BEST.pt"),
    ),
}


def _1x1_layers(net, model: str):
    """The two fresh 1x1 conv layers, wherever `add1x1front`/`add_1x1_front`
    nested them for this model (different per model, no shared base class)."""
    if model == "Dist U-Net":
        return net.model.new_layer_1, net.model.new_layer_2
    if model == "Cellpose":
        return net.net.downsample[0], net.net.downsample[1]
    if model == "Stardist":
        return net.encoder[0], net.encoder[1]
    if model == "InstanSeg":
        return net.backbone[0], net.backbone[1]
    raise ValueError(f"no 1x1 extraction wired for {model!r}")
