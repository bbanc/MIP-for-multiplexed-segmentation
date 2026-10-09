from dataclasses import dataclass
import warnings

# Cellpose warning
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*", category=UserWarning)

from typing import Any, Callable, Dict

import numpy as np


@dataclass
class ModelSpec:
    make_model: Callable[[Dict], Any]
    infer: Callable[[Any, np.ndarray, Dict], np.ndarray]


try:
    from segment_custom import CUSTOM_MODELS
except ImportError:
    CUSTOM_MODELS = {}

def _make_cellpose(hyper: Dict) -> Any:
    from cellpose import models

    model_type = hyper.get("model_type", None)
    gpu = bool(hyper.get("use_gpu", True))
    model_path = hyper.get("model_path", None)

    if model_path in (None, "None", ""):
        return models.CellposeModel(model_type=model_type, gpu=gpu)
    return models.CellposeModel(model_type=model_type, gpu=gpu, pretrained_model=model_path)


def _infer_cellpose(model: Any, img: np.ndarray, hyper: Dict) -> np.ndarray:
    flow_threshold = hyper.get("flow_threshold", None)
    cellprob_threshold = hyper.get("cellprob_threshold", None)
    masks, _, _ = model.eval(
        img,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks


MODELS: Dict[str, ModelSpec] = {
    "cellpose": ModelSpec(make_model=_make_cellpose, infer=_infer_cellpose),
    "cpose": ModelSpec(make_model=_make_cellpose, infer=_infer_cellpose),
}

if isinstance(CUSTOM_MODELS, dict):
    MODELS.update({name.lower(): spec for name, spec in CUSTOM_MODELS.items()})

def _get_spec(model_name: str) -> ModelSpec:
    name = (model_name or "cellpose").lower()
    spec = MODELS.get(name)
    if spec is None:
        raise ValueError(f"Unknown segmentation model: {model_name}")
    return spec


def make_model(model_name: str, hyper: Dict) -> Any:
    """Return a model object for inference."""
    return _get_spec(model_name).make_model(hyper)


def infer(model_name: str, model: Any, img: np.ndarray, hyper: Dict) -> np.ndarray:
    """Run inference for a single image."""
    return _get_spec(model_name).infer(model, img, hyper)


def segment(img: np.ndarray, model_name: str, hyper: Dict) -> np.ndarray:
    model = make_model(model_name, hyper)
    return infer(model_name, model, img, hyper)
