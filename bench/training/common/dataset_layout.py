"""Per-dataset directory layout: which folders on disk hold a given sample.

This is the "which folders match this dataset name" half of each
Experiments script's get_dataset_paths — identical across every model for a
given dataset, since it reflects how that dataset was exported to disk, not
anything model-specific. File-naming within a folder (raw tif vs. precomputed
ChannelNet png vs. fixed MACSima filenames) still varies by preprocessing
variant and stays local to each script.
"""
import os
import glob


def codex_layout(base_path, name):
    return sorted(glob.glob(os.path.join(base_path, f"CODEX_{name}*/")))


def vectra_layout(base_path, name):
    return sorted(glob.glob(os.path.join(base_path, name, "*/")))


def macsima_layout(base_path, name):
    return sorted(glob.glob(os.path.join(base_path, f"{name}*/")))


def zeiss_layout(base_path, name):
    return glob.glob(os.path.join(base_path, f"{name}*/"))
