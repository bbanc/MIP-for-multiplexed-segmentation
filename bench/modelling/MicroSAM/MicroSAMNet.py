import torch
from micro_sam.automatic_segmentation import get_predictor_and_segmenter


class MicroSAMModel:
    def __init__(self, predictor, segmenter, device):
        self.predictor = predictor
        self.segmenter = segmenter
        self.device = device


def build_pretrained(model_type="vit_b_lm", device=None, segmentation_mode="ais"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, segmentation_mode=segmentation_mode, device=str(device),
    )
    return MicroSAMModel(predictor, segmenter, device)
