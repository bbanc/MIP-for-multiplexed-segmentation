from modelling.common.AdaptorNetBase import AdaptorNetSegmenterBase
from modelling.Cellvit.StarDistRN50_Rnd import StarDistRN50_Rnd


class AdaptorNetStardist(AdaptorNetSegmenterBase):
    def calculate_instance_map(self, *args, **kwargs):
        return self.segmenter.calculate_instance_map(*args, **kwargs)

    def compute_loss(self, x, targets):
        raw = self(x)  # gradients flow: loss -> segmenter -> adaptor_net
        return self.segmenter.loss_fn(raw, targets)


def build(pretrained_path, n_seg_cls=1, n_rays=32, device="cpu", adaptor_kwargs=None):
    segmenter = StarDistRN50_Rnd(n_rays=n_rays, n_seg_cls=n_seg_cls, device=device)
    segmenter.load_model(pretrained_path)
    return AdaptorNetStardist(segmenter=segmenter, adaptor_kwargs=adaptor_kwargs)
