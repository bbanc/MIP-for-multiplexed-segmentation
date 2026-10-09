from modelling.common.AdaptorNetBase import AdaptorNetSegmenterBase
from modelling.UNet.DistNet import DIST_Net_Lightning


class AdaptorNetDIST(AdaptorNetSegmenterBase):
    """AdaptorNet (channel-reduction front end) + DIST's pretrained 3-channel
    U-Net segmenter, trained end-to-end on DIST's own binary+distance loss.

    Replaces `DistNet.py`'s `add_1x1_front` (two fresh, randomly-initialized
    1x1 convs) for the 1x1/multiplexed_input experiment variants -- AdaptorNet
    is a channel-invariant architecture purpose-built for this reduction,
    rather than a fixed-width ad hoc adapter.
    """

    @property
    def model(self):
        """Alias so `model.model(x)` -- the call `bench/inferencing/DIST_inference.py`
        and every DIST Experiments script already use -- triggers the composed
        adaptor_net -> segmenter forward, without touching that shared file
        (used by 16 existing UNet Experiments scripts) at all."""
        return self

    def compute_loss(self, x, targets):
        raw = self(x)  # gradients flow: loss -> segmenter -> adaptor_net
        return self.segmenter.loss_fn(raw, targets)


def build(pretrained_path, arch="unet", encoder_name="resnet50", out_classes=2, adaptor_kwargs=None):
    """Fresh AdaptorNet in front of DIST's pretrained (Stage 2) 3-channel U-Net."""
    segmenter = DIST_Net_Lightning.load_from_checkpoint(
        arch=arch, encoder_name=encoder_name, in_channels=3, out_classes=out_classes,
        checkpoint_path=pretrained_path,
    )
    return AdaptorNetDIST(segmenter=segmenter, adaptor_kwargs=adaptor_kwargs)
