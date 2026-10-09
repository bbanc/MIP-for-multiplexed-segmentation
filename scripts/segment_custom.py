"""Custom segmentation model.

To add a custom segmentation algorithm:

1. Pick a model name, e.g. "MY_NEW_MODEL".
2. Implement `_make_MY_NEW_MODEL` and `_infer_MY_NEW_MODEL`.
3. Register the model in `CUSTOM_MODELS` at the bottom of this file.
4. Add a corresponding configuration block under
   `segmentation.hyperparameters.MY_NEW_MODEL` in `config/config.yaml`.
5. Set `segmentation.model: MY_NEW_MODEL` in `config/config.yaml`.

"""

from typing import Any, Dict

import numpy as np
from scripts.segment_models import ModelSpec


def _make_MY_NEW_MODEL(hyper: Dict) -> Any:
    """Example: build/load a custom model.

    Parameters
    ----------
    hyper : Dict
        Hyperparameters for this model, configured in config/config.yaml under
        segmentation.hyperparameters.MY_NEW_MODEL.

    Returns
    -------
    Any
        Model object that will be passed to `_infer_MY_NEW_MODEL`.
    """

    # Example pseudocode:
    #   from mymodel import MyNet
    #   model_path = hyper.get("model_path", "/path/to/model.pt")
    #   model = MyNet.load_from_checkpoint(model_path)
    #   model.eval()
    #   return model
    raise NotImplementedError("_make_MY_NEW_MODEL not implemented.")


def _infer_MY_NEW_MODEL(model: Any, img: np.ndarray, hyper: Dict) -> np.ndarray:
    """Example: run inference with a custom model.

    Parameters
    ----------
    model : Any
        Model instance from `_make_MY_NEW_MODEL`.
    img : np.ndarray
        Input image as provided by the pipeline. Shape is (H, W, C), dtype is uint8. 
    hyper : Dict
        Same hyperparameter dict as for `_make_MY_NEW_MODEL`.

    Returns
    -------
    np.ndarray
        Labelled mask of shape (H, W) with integer labels, background = 0,
        and foreground cells labelled 1..N.
    """

    # Example pseudocode:
    #   logits = model(img[None, ...], threshold=hyper.get("threshold", 0.5))
    #   masks = postprocess_to_labels(logits) - to be implemented
    #   return masks.astype(np.int32)
    raise NotImplementedError("_infer_MY_NEW_MODEL not implemented.")

CUSTOM_MODELS: Dict[str, ModelSpec] = {
    # "MY_NEW_MODEL": ModelSpec(
    #     make_model=_make_MY_NEW_MODEL,
    #     infer=_infer_MY_NEW_MODEL,
    # ),
}
