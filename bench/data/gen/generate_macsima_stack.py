"""Generate stacked_img.tif / protein_names.csv / Annotation.png for raw MACSima exports.

Reimplements the old images2stack.ipynb notebook as a script, using the same
Dataset-class conventions as MultiplexedDataset (Stacked_Multiplexed.py):
RawMACSimaChannelDataset maps each raw dataset folder to its per-marker TIFFs
and stacks them, DAPI first, instead of the notebook's inline loops.

Raw MACSima layout expected under Multiplex/MACSima/:
    <Dataset>_.../<ROI folder>/*.tif   - one raw single-channel TIFF per marker,
                                          DAPI identified by "dapi" in the filename
    <Dataset>_.../Annot.png            - raw label mask sibling of the ROI folder

Every model's Experiments/MACSima/*.py script (and generate_channelnet_rgb.py)
expects stacked_img.tif and Annotation.png to already sit in <Dataset>_.../,
which is what this script produces.

Channel ordering (2026-08-21 rewrite): every dataset group covered by
`channels.MACSIMA_SELECTED_MARKERS` gets a CANONICAL channel order --
DAPI always at slot 0, that group's curated markers at fixed slots 1..N in the
same order for every sample, remaining real markers after that in arbitrary
order. This makes raw indices stable per group (previously they weren't: the
same marker could land at a different index in different samples, since panel
composition varies slightly sample to sample), so downstream loading code
(`training/common/channels.py`) can go back to a single static index list per
group instead of resolving marker names per image at load time -- run this
script once, and everything downstream is simple again.

Also fixes a DAPI bug this rewrite uncovered: MACSima's cyclic imaging protocol
re-stains DAPI at the end of some panels (e.g. Liver, TNBC each have two files
with "dapi" in the name -- an initial stain plus a much-later re-stain cycle).
The old code took the first one found; the cached MaxProj.png files this
pipeline has always treated as ground truth were verified (pixel-exact match)
to be built from the LAST one instead. This script now does the same, and
drops the earlier, redundant DAPI acquisition entirely rather than keeping it
as an ordinary channel -- so there is never a second channel also named
"DAPI" for downstream code to trip over.
"""
import argparse
import glob
import os
import re
import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import numpy as np
import pandas as pd
import skimage.io
from torch.utils.data import Dataset

from import_util import data_path
from training.common.channels import MACSIMA_SELECTED_MARKERS

PROTEIN_NAME_RE = re.compile(r"ROI-\d+_A-(.*?)(?:_C-|\.tif)")


def _protein_name(tif_path):
    name = os.path.basename(tif_path)
    match = PROTEIN_NAME_RE.search(name)
    return match.group(1) if match else os.path.splitext(name)[0]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _roi_folder(dataset_folder):
    subdirs = glob.glob(os.path.join(dataset_folder, "*/"))
    if len(subdirs) != 1:
        raise ValueError(f"Expected exactly one ROI subfolder in {dataset_folder}, found {len(subdirs)}")
    return subdirs[0]


def _group_for(dataset_folder):
    """Which MACSIMA_SELECTED_MARKERS group this sample belongs to, if any."""
    name = os.path.basename(os.path.normpath(dataset_folder))
    for group in MACSIMA_SELECTED_MARKERS:
        if name.startswith(group):
            return group
    return None


class RawMACSimaChannelDataset(Dataset):
    """Maps each raw MACSima dataset folder to its per-marker TIFFs, DAPI first."""

    def __init__(self, dataset_folders):
        self.dataset_folders = dataset_folders

    def load_stack(self, idx):
        """(C, H, W) stack + protein names: correct DAPI first, then (for verified
        groups) the canonical marker order, then everything else."""
        folder = self.dataset_folders[idx]
        roi_folder = _roi_folder(folder)
        img_paths = sorted(glob.glob(os.path.join(roi_folder, "*.tif")))

        dapi_paths = [p for p in img_paths if "dapi" in p.lower()]
        if not dapi_paths:
            raise ValueError(f"Expected at least one DAPI channel in {roi_folder}, found none")
        dapi_path = dapi_paths[-1]  # last acquisition = the one MaxProj.png was built from
        other_paths = [p for p in img_paths if p not in dapi_paths]  # drop earlier DAPI(s) too

        group = _group_for(folder)
        if group is not None:
            wanted = MACSIMA_SELECTED_MARKERS[group]
            by_name = {_norm(_protein_name(p)): p for p in other_paths}
            missing = [m for m in wanted if _norm(m) not in by_name]
            if missing:
                raise ValueError(f"{roi_folder}: canonical {group} markers not found: {missing}")
            selected = [by_name[_norm(m)] for m in wanted]
            rest = [p for p in other_paths if p not in selected]
            other_paths = selected + rest

        ordered_paths = [dapi_path] + other_paths
        protein_names = ["DAPI"] + [_protein_name(p) for p in other_paths]

        stack = np.stack([skimage.io.imread(p) for p in ordered_paths], axis=0)
        return stack, protein_names

    def __len__(self):
        return len(self.dataset_folders)

    def __getitem__(self, idx):
        return self.load_stack(idx)


def generate(base_path, overwrite=False):
    dataset_folders = sorted(glob.glob(os.path.join(base_path, "*", "")))
    dataset = RawMACSimaChannelDataset(dataset_folders)

    for i, folder in enumerate(dataset_folders):
        stack_path = os.path.join(folder, "stacked_img.tif")
        if overwrite or not os.path.exists(stack_path):
            stack, protein_names = dataset.load_stack(i)
            skimage.io.imsave(stack_path, stack, check_contrast=False)
            pd.DataFrame({
                "Index": range(len(protein_names)),
                "Protein Name": protein_names,
            }).to_csv(os.path.join(folder, "protein_names.csv"), index=False)
            print(f"wrote {stack_path}")
        else:
            print(f"skip (exists): {stack_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs that already exist")
    args = parser.parse_args()
    generate(str(data_path("Multiplex/MACSima/")), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
