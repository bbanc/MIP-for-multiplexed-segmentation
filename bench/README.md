# Benchmarking and training code

This holds the benchmarking code to train and evaluate models and collect the results to produce paper figures.

Note: The Snakemake pipeline to *use* MIP is in the main  (see the main [README](../README.md)).

---

## Summary

**Models**

| Model | Folder | Run-name prefix | Trained here? |
|---|---|---|---|
| Dist U-Net | `UNet` | `DIST` | yes |
| Stardist (CellViT) | `CellViT_Star` | `Stardist` | yes |
| Cellpose | `Cellpose` | `CPose2ch` | yes |
| InstanSeg | `InstanSeg` | `InstanSeg` | yes |
| Mesmer (DeepCell) | `Mesmer` | `Mesmer` | yes (Max-Proj only) |
| Cellpose-SAM | `Cellpose_SAM` | `CPoseSAM_ZeroShot` | no, zero-shot (Max-Proj only) |
| MicroSAM | `MicroSAM` | `MicroSAM_ZeroShot` | no, zero-shot (Max-Proj only) |

**Datasets**: Vectra (132 images), CODEX (10), MACSima (89) and Zeiss (19). Every model is scored with 5-fold cross-validation (`KFold(5, shuffle=True, random_state=42)`). All models see exactly the same train and test images.

**Input methods** (for each model in `./training/[MODEL]/Experiments/Datasets`):

| Script | Meaning |
|---|---|
| `run_max_proj.py` | Max-projection of multiplexed channels to 3 channels (MIP) |
| `run_1x1.py` | Two 1x1 conv layers in front of the backbone for learned linear combinations (LLC) |
| `run_multiplexed_input.py` | The backbone's first layer suited for multiplexed adapted model (MAM) |
| `run_channelnet_tuned.py` | ChannelNet's AdaptorNet in front of the backbone for each model |

**Channel arms** (`training/common/channels.py` is the single source of truth):

- `selected`: DAPI plus up to 7 curated markers, always 8 channels (DAPI first; unused slots are zero).
- `all`: every marker in the raw stack, zero-padded to the dataset's largest panel (Vectra 8, CODEX 32, MACSima 101, Zeiss 7).

**Pretraining** (all trained models start from the same two stages, per model): Stage 1 on TissueNet + LIVECell (`Stage_1/Train_Stage1.py`), Stage 2 on the Cellpose dataset (`Stage_2/Train_Stage2.py`). Experiments load the Stage 2 checkpoint (`PRETRAINED_PATH` in each script).

---

## Reproducing figures

`all_runs.csv` has one row per (model, dataset, input method, arm, fold, test image): the scores, the sweep columns, the prediction's shape statistics (`Area`, `Solidity`, ...) and the shape of the ground truth (`GT_*`). It picks `_selected` over `_filtered` when both exist. Every notebook below reads it.

| Notebook (`figures/`) | Makes |
|---|---|
| `Input_Method_Comparison` | Input method comparisons |
| `N_Channels_Comparisons` | `selected` vs. `all` channels |
| `Model_Comparison` | The 7 models on MIP |
| `Example_Segmentations_and_Shape_Bias` | Example segmentations and shape biases |
| `Performance_by_Cell_Shape` | Performance depending on cell shape |
| `Consensus_Performance` | Which cells do all models detect? How well are they segmented? |
| `Adapter_Outputs`* | Example outputs for each preprocessing method |
| `Processing_Time`* | Processing time per method on CPU and GPU |

The notebooks used to generate figures are in ./figures. They load `./results/all_runs.csv` and write to `figures/export/`.

\* Requires trained models.

---

## Layout

```
bench/
  import_util.py     paths to data and CellViT repository
  data/              dataset classes & definitions
  modelling/         Model implementations & training loops for models and ChannelNet (AdaptorNet), one folder per model
  inferencing/       network output -> labelled mask, one file per model
  training/
    common/          channels.py (channel selection), dataset_layout.py, pipeline.py 
    {Model}/Stage_1, Stage_2/       pretraining
    {Model}/Experiments/{Dataset}/  one script per input method
  validation/        metrics
  misc/              
  figures/           notebooks to make the figures
  results/           all_runs.csv
```

---

## Setup

1. **Paths.** Edit `DATA_ROOT` in `import_util.py`. Expected under `DATA_ROOT`: `Multiplex/{Vectra,CODEX,MACSima,Zeiss}/`, `TissueNet/`, `LIVECell/`, `Cellpose/`. If using Stardist/CellViT: Set the `CELLVIT_ROOT` environment variable to checkout folder of CellViT.
2. **Prepare the data.** MACSima raw exports are turned into `stacked_img.tif`, `protein_names.csv` and `Annotation.png` by `data/gen/generate_macsima_stack.py`.

### Environments

Everything except Mesmer ran in two [uv](https://docs.astral.sh/uv/) virtual environments. Mesmer needs TensorFlow 2.8 with CUDA 11 and cuDNN 8, which our host did not have, so it ran in a Docker container. The requirements files list the versions we used.

| Environment | Python | Used for | Requirements |
|---|---|---|---|
| `.venv` | 3.12 | Dist U-Net, Stardist, Cellpose, InstanSeg, MicroSAM | `requirements.txt` |
| `.venv_cpsam` | 3.12 | Cellpose-SAM (needs cellpose 4.x, conflicts with Cellpose 3.1 in `.venv`) | `requirements_cpsam.txt` |
| Docker image `mesmer-gpu` | 3.8 | Mesmer (tensorflow 2.8.4, deepcell 0.12.10) | `requirements_mesmer.txt` |

```bash
# from the repository root
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r bench/requirements.txt

uv venv .venv_cpsam --python 3.12
uv pip install --python .venv_cpsam/bin/python -r bench/requirements_cpsam.txt
```

Install the CUDA build of torch that matches your driver (see the [PyTorch install guide](https://pytorch.org/get-started/locally/)). 

**Mesmer (Docker).** We ran Mesmer using the official Docker container from [DeepCell-TF](https://github.com/vanvalenlab/deepcell-tf) ("Install with Docker"). Exact versionings are documented in `bench/requirements_mesmer.txt`. 

---

## Running

**Pretraining.** For each trained model run `training/{Model}/Stage_1/Train_Stage1.py`, then `training/{Model}/Stage_2/Train_Stage2.py`. Experiments then load the Stage 2 checkpoint.

**A single experiment**, for example Dist U-Net, 1x1 adapter, Vectra, selected arm, fold 0:

```bash
python bench/training/UNet/Experiments/Vectra/run_1x1.py --arm selected --fold 0
```

Omit `--arm` / `--fold` to run everything.

**Outputs**, under `training/{Model}/Experiments/{Dataset}/`:
- `models/`, `saves/` or `runs/`: artifacts.
- `scores_n/{run}_fold{k}/`: `scores.csv` (per-image Prec, Rec, F1, mIoU, mDICE, PQ), `shapes.csv` (per-image shape statistics of the prediction), `sweep.csv` (precision/recall/F1 at IoU thresholds 0.1-1.0) and `preds/` (the predicted label images).
- Runs are named `{Model}_{Dataset}_{Method}_fold{k}`, e.g. `DIST_Vectra_1x1_selected_fold0`. 
