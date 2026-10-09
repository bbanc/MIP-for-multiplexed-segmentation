# Cell segmentation for multiplexed images

This repository provides a reproducible [Snakemake](https://snakemake.readthedocs.io/) pipeline for cell segmentation and feature extraction from multiplexed microscopy images. It handles

- image normalization
- preprocessing of multiplexed data
- cell segmentation
- postprocessing of cell masks
- single-cell feature extraction
- optional evaluation against human-annotated ground truth

Additionally, it contains the reference code for model evaluation performed as part of the paper publication under [`./bench`](bench/)

---

## Reference

[T.B.D.] 

---

## Installation

### 1) Create an environment

The pipeline runs in a single Python environment (tested with Python 3.12). With [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv_pipeline --python 3.12
source .venv_pipeline/bin/activate
uv pip install -r requirements.txt
```

### 2) GPU-support

The pipeline uses [Cellpose](https://github.com/MouseLand/cellpose) segmentation by default. To enable GPU support, install the CUDA build of PyTorch that matches your driver before installing the requirements, as described in the [Cellpose installation guide](https://cellpose.readthedocs.io/en/latest/installation.html).

--- 

## Running and configuring

### Run

After installation, the pipeline may then be launched (from the repository root) via:

```bash
snakemake -n
```
This performs a dry-run: it lists the steps that would run, without running anything.

Then, to fully run the pipeline:

```bash
snakemake --cores 8
```

See the [Snakemake usage notes](https://snakemake.readthedocs.io/en/stable/executable.html) for additional usage notes, flags, and commands.


### Configuring
The pipeline is controlled via [`./config/config.yaml`](config/config.yaml). It holds a commented reference configuration. The main components are:

- `samples`: define sample IDs and data paths (and optional nuclear channels via `dapi_channel`).
- `segmentation`: Segmentation parameters and model hyperparameters.
- `evaluation` (optional): If ground truth masks are available, these may be passed here to test performance.

The pipeline writes outputs to `results/{sample}/`.


### Outputs

For each sample, the pipeline creates:

- `results/{sample}/max_proj.png` - The max-projected multiplexed image.
- `results/{sample}/mask_final.tif` - labelled cell segmentation masks. 
- `results/{sample}/cell_features.csv` - per-cell feature table.
- `results/{sample}/cell_features.h5ad`  - features in AnnData format for downstream analysis.
- `results/{sample}/evaluation.csv` (if evaluation enabled) - evaluation against ground thruth (if available).

---

## Notebook runtime
We additionally provide an interactive notebook [`Pipeline_Demo.ipynb`](Pipeline_Demo.ipynb) outlining the main analysis steps. It runs on the small tonsil example in [`example_data/`](example_data/). 

---

## Advanced Usage

### Adding additional segmentation methods  

The pipeline is modular by design and additional segmentation methods can be added via the following steps:

#### 1) Add model wrapper code in `scripts/segment_custom.py`

This requires two functions:

- `*_make_<MY_NEW_MODEL>()`: to build/load the model
- `*_infer_<MY_NEW_MODEL>()`: to run inference on a single image and return a labelled mask (numbered 1..N)

Then, add the new functions in the `MODELS` variable through the ModelSpec dataclass.

Example:

```python
def _make_MY_NEW_MODEL(hyper: Dict) -> Any:
    # load your model here
    return model

def _infer_MY_NEW_MODEL(model: Any, img: np.ndarray, hyper: Dict) -> np.ndarray:
    # run inference here
    return masks

MODELS["MY_NEW_MODEL"] = ModelSpec(make_model=_make_MY_NEW_MODEL, infer=_infer_MY_NEW_MODEL)
```

Notes:
- `infer` must return a contiguous **labelled mask** with background = 0.
- `hyper` is a dict containing all hyperparameters defined in `config/config.yaml`. It gets passed to both the make_model and infer funcs.

#### 2) Add hyperparameters in `config/config.yaml`

Use one block per model, for example:

```yaml
segmentation:
  model: MY_NEW_MODEL
  hyperparameters:
    MY_NEW_MODEL:
      thresh: 0.5
      model_path: "/PATH/TO/MODEL.pt"
```

#### 3) Ensure dependencies

If the new model needs extra packages, install them into the pipeline environment and add them to `requirements.txt`.

### Useful Snakemake commands

1) The pipeline can be stopped by specifying the final output e.g.:

Stop after segmentation mask for sample1:

```bash
snakemake results/sample1/mask_final.tif
```

Only features for Mynewsample:
```bash
snakemake results/Mynewsample/cell_features.csv
```

2) The -c (or -j) flag handles how many jobs Snakemake can parallelize when running multiple samples.  