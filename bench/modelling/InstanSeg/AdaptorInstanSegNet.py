from modelling.common.AdaptorNetBase import AdaptorNetSegmenterBase
from modelling.InstanSeg.InstanSegNet import build_fresh


class AdaptorNetInstanSeg(AdaptorNetSegmenterBase):
    def forward(self, x):
        x3 = self.adaptor_net(x)
        return self.segmenter.backbone(x3)

    def compute_loss(self, x, targets):
        raw = self(x)  # loss: segmenter.backbone -> adaptor_net
        return self.segmenter.method.forward(raw, targets.clone()).mean()

    def set_loss_stage(self, stage: str):
        if stage == "hotstart": # From InstanSeg
            self.segmenter.method.update_seed_loss("binary_xloss")
            self.segmenter.method.update_binary_loss("dice_loss")
        elif stage == "main":
            self.segmenter.method.update_seed_loss("l1_distance")
            self.segmenter.method.update_binary_loss("lovasz_hinge")
        else:
            raise ValueError(stage)

    def to(self, device):
        super().to(device)  
        self.segmenter.method.device = device
        return self

    @property
    def backbone(self):
        return self

    @property
    def method(self):
        return self.segmenter.method


def build(pretrained_path, device="cpu", adaptor_kwargs=None):
    """Fresh AdaptorNet in front of InstanSeg's pretrained (Stage 2) 3-channel backbone."""
    segmenter = build_fresh(device=device)
    segmenter.load_model(pretrained_path, device=device)
    return AdaptorNetInstanSeg(segmenter=segmenter, adaptor_kwargs=adaptor_kwargs)
