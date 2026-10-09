import torch

from modelling.common.AdaptorNetBase import AdaptorNetSegmenterBase
from modelling.Cellpose.myCellpose import myCellpose


class AdaptorNetCellpose(AdaptorNetSegmenterBase):
    def __init__(self, cellpose_model, adaptor_kwargs=None):
        super().__init__(segmenter=cellpose_model.net, adaptor_kwargs=adaptor_kwargs or {"out_channels": 2})
        self._cellpose_model = cellpose_model

    def forward(self, x):
        x3 = self.adaptor_net(x)
        return self.segmenter(x3)  # (y, style)

    def compute_loss(self, x, targets):
        y = self(x)[0]  # gradients flow: loss -> segmenter -> adaptor_net
        return self._cellpose_model.loss_fn(targets, y)

    def to(self, device):
        super().to(device)
        self._cellpose_model.device = torch.device(device)
        return self

    @property
    def net(self):
        """Alias so `model.net(imgs)` triggers the composed adaptor_net -> segmenter forward."""
        return self

    def _compute_masks(self, *args, **kwargs):
        return self._cellpose_model._compute_masks(*args, **kwargs)


def build(pretrained_path, adaptor_kwargs=None, gpu=True):
    """Fresh AdaptorNet (2-channel output) in front of Cellpose's pretrained CPnet."""
    cellpose_model = myCellpose(nchan=2, gpu=gpu, pretrained_model=pretrained_path)
    return AdaptorNetCellpose(cellpose_model, adaptor_kwargs=adaptor_kwargs)
