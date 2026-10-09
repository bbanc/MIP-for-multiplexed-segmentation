import glob
import os

import numpy as np

from data.Stacked_Multiplexed import max_project
from training.common.dataset_layout import (
    codex_layout, macsima_layout, vectra_layout, zeiss_layout,
)

MAX_PANEL = {
    "Vectra": 8,     # uniform across all 132 images
    "CODEX": 32,     # LN 28, Tnsl 32
    "MACSima": 101,  # Tonsils 30-31, OvCa_1 40, OvCa_3 54, OvCa_2 56, TNBC 69, Liver 101
    "Zeiss": 7,      # e.g. ZP-10002
}

# Where are the nuclei (DAPI)?
VECTRA_DAPI = {
    "P01": 5, "P02": 6, "P03": 6, "P04": 5, "P05": 5, "P06": 5,
    "P07": 5, "P08": 5, "P09": 6, "P10": 6, "P11": 5, "P12": 6,
    "P13": 6, "P14": 6, "P15": 5, "P16": 6,
}
ZEISS_DAPI = {"ZP-10002": 2}

# Vectra has an autofluorescence channel.
VECTRA_AF = 7


def dapi_index(dataset, name):
    """Raw DAPI channel index for one dataset entry."""
    if dataset == "Vectra":
        return VECTRA_DAPI[name]
    if dataset == "Zeiss":
        return ZEISS_DAPI.get(name, 0)
    return 0 # MACSima/CODEX: 0

# Visually chosen
CODEX_SELECTED = {
    "LN": [0, 17, 22, 18, 16, 19, 23],
    "Tnsl": [0, 16, 26, 12, 19, 27],
}

MACSIMA_SELECTED_MARKERS = {
    "Tonsils": ["HLA-DR", "CD44", "CD3", "CD11c", "Vimentin", "PlasmaCell", "CD20 Cytoplasmic"],
    "TNBC": ["CD8a", "CD4", "HLA-DR", "CD31", "CD45RO", "CD44", "Cytokeratin"],
    "OvCa_1": ["CD326", "Cytokeratin7", "CD11c", "CD4", "CD8", "CD45", "CD45RA"],
    "OvCa_2": ["CD326", "Cytokeratin7", "CD44", "CD11c", "CD4", "CD8", "CD45RA"],
    "OvCa_3": ["CD326", "Cytokeratin7", "CD44", "CD11c", "CD4", "CD8", "CD45"],
    "Liver": ["Hepatocyte", "CD45RO", "CD79a", "CD4", "HLA-DR", "CD138", "CD163"],
}

# Visually chosen
ZEISS_SELECTED = {
    "ZP-10001": (0, 4, 1, None, None, 2, None),
    "ZP-10002": (2, 0, 4, 6, 3, 1, None),
    "PDAC": (0, 4, None, 1, 2, 3, None),
    "Spleen": (0, 4, None, 1, 2, 3, None),
    "ZP-9999": (0, 4, None, 1, None, None, 2),
}

def selected_indices(dataset, name):
    if dataset == "Vectra": # Take all but AF
        dapi = dapi_index(dataset, name)
        rest = [i for i in range(MAX_PANEL["Vectra"]) if i not in (dapi, VECTRA_AF)]
        return [dapi] + rest
    if dataset == "MACSima":  # DAPI, then the markers in the order generate_macsima_stack.py writes them
        idxs = list(range(len(MACSIMA_SELECTED_MARKERS[name]) + 1))
    else:
        idxs = list({"CODEX": CODEX_SELECTED, "Zeiss": ZEISS_SELECTED}[dataset][name])
    max_slots = 8
    if len(idxs) > max_slots:
        raise ValueError(f"{dataset}/{name}: selected arm has {len(idxs)} slots, max is {max_slots}")
    if idxs[0] is None or idxs[0] != dapi_index(dataset, name):
        raise ValueError(f"{dataset}/{name}: selected arm must start with the raw DAPI index "
                         f"{dapi_index(dataset, name)}, got {idxs[0]}")
    return idxs


def all_indices(dataset, name, n_available):
    """Raw channel indices for the "all" arm: DAPI first, then every other channel."""
    dapi = dapi_index(dataset, name)
    return [dapi] + [i for i in range(n_available) if i != dapi]

METHODS = ("max_proj", "1x1", "multiplexed_input", "channelnet_tuned")
ARMS = ("selected", "all")


def n_input_channels(dataset, arm, method):
    if method == "max_proj":
        return 3
    if arm == "selected":
        return 8  # DAPI + up to 7 markers; unused slots are zero (see _gather)
    return MAX_PANEL[dataset]


def image_filename(dataset, arm, method):
    if dataset != "MACSima":
        return None
    if arm == "selected" and method == "max_proj":
        return "MaxProj.png"
    return "stacked_img.tif"


def _gather(stack, idxs, width):
    """Select `idxs` from a raw HWC stack into a `width`-channel array, zero-filling gaps."""
    h, w, c = stack.shape
    out = np.zeros((h, w, width), dtype=stack.dtype)
    for slot, idx in enumerate(idxs[:width]):
        if idx is not None and 0 <= idx < c:
            out[:, :, slot] = stack[:, :, idx]
    return out


def make_stack_func(dataset, name, arm, method):
    if method not in METHODS:
        raise ValueError(f"unknown input method {method!r}, expected one of {METHODS}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    if dataset == "MACSima" and arm == "selected" and method == "max_proj":
        return lambda stack: stack  # the cached MaxProj.png is already the projection

    width = n_input_channels(dataset, arm, method)
    idxs = selected_indices(dataset, name) if arm == "selected" else None

    def stack_func(stack):
        i = idxs if idxs is not None else all_indices(dataset, name, stack.shape[-1])
        if method == "max_proj":
            return max_project(_gather(stack, i, len(i)))
        return _gather(stack, i, width)
    return stack_func


def build_paths(dataset, dataset_names, base_path, arm, method):
    layout = {  # which folders on disk belong to a sample name
        "Vectra": vectra_layout,
        "CODEX": codex_layout,
        "MACSima": macsima_layout,
        "Zeiss": zeiss_layout,
    }[dataset]
    img_paths, label_paths, funcs = [], [], []

    for name in dataset_names:
        stack_func = make_stack_func(dataset, name, arm, method)

        for folder in layout(base_path, name):
            if dataset == "MACSima":
                img_paths.append(os.path.join(folder, image_filename(dataset, arm, method)))
                label_paths.append(os.path.join(folder, "Annotation.png"))
            else:
                img_paths.append(glob.glob(os.path.join(folder, "*-Crop_Tif.tif"))[0])
                label_paths.append(glob.glob(os.path.join(folder, "*-Crop_Tif_LABELS.png"))[0])

            funcs.append(stack_func)

    return (np.array(img_paths), np.array(label_paths), np.array(funcs),
            np.array([None] * len(img_paths)))
