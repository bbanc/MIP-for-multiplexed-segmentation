import torch.nn as nn
import lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
import torch
import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

class DIST_Net_Lightning(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        #self.loss_binary = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.loss_binary = nn.BCEWithLogitsLoss()
        self.loss_distance = nn.MSELoss()
        # Set by the caller (Experiments train_model()) before trainer.fit() so
        # configure_callbacks() below knows where/what to name the best-val-loss
        # checkpoint. Left unset, no ModelCheckpoint is attached (Trainer behaves
        # exactly as before this attribute existed).
        self.checkpoint_dir = None
        self.checkpoint_name = None
        # Overridable by the caller the same way (model.lr = LEARNING_RATE) --
        # Lightning calls configure_optimizers() with zero arguments, so a plain
        # parameter default there is silently unreachable from the outside.
        self.lr = 1e-3

    def forward(self, image):
        labels = self.model(image)
        return labels

    def training_step(self, batch):
        inputs, target = batch
        output = self(inputs,)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def configure_callbacks(self):
        """Lightning merges callbacks returned here into the Trainer automatically on
        `trainer.fit(model, ...)` -- regardless of what callbacks (if any) the Trainer
        itself was constructed with. Makes early stopping the default for every
        Experiments script without needing to touch any of their `pl.Trainer(...)`
        calls; matches the patience UNet's own Stage_1/Stage_2 pretraining already uses
        the pattern of (there it's passed explicitly, with patience=15).

        Also attaches a ModelCheckpoint saving the best val_loss epoch as
        `{checkpoint_name}_BEST.ckpt` -- matching the `{model_name}_BEST.*` convention
        every other model family (Cellpose/CellViT_Star/InstanSeg/AdaptorNetBase) already
        uses -- whenever the caller has set `self.checkpoint_dir`/`self.checkpoint_name`
        (see Experiments train_model()). EarlyStopping alone never saved a checkpoint;
        without this, "early stopping" only ever meant "stop sooner", not "keep the best
        weights" -- callers that don't set these two attributes get exactly the old
        behavior (EarlyStopping only, no ModelCheckpoint)."""
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=25)]
        if self.checkpoint_dir and self.checkpoint_name:
            from lightning.pytorch.callbacks import ModelCheckpoint
            callbacks.append(ModelCheckpoint(
                dirpath=self.checkpoint_dir, filename=f"{self.checkpoint_name}_BEST",
                monitor="val_loss", mode="min", save_top_k=1,
            ))
        return callbacks

    def validation_step(self, batch):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss_fn(output, target)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def predict_step(self, batch):
        inputs, _ = batch
        with self.eval():
            out = self.model(inputs)
        return out

    def inference_step(self, img, mask_thresh=0.5, min_peak_dist=10):
        from skimage.feature import peak_local_max
        from skimage.measure import label
        from skimage.segmentation import watershed


        img_input, pad_h, pad_w = prepare_input(img)
        out = self.model(img_input).detach()
        
        binary_prob = out[:,0,:, :].sigmoid()
        dist_reg = out[:,1,:,:]
        
        mask = (binary_prob[0,] > mask_thresh).cpu().numpy()
        dist = (dist_reg[0,]).cpu().numpy()

        mask = crop_to_original(mask, pad_h, pad_w)
        dist = crop_to_original(dist, pad_h, pad_w)

        peak_coords = peak_local_max(dist, min_distance=min_peak_dist, labels=mask)
        
        seeds_bin = np.zeros_like(mask, dtype=bool)
        seeds_bin[tuple(peak_coords.T)] = 1
        
        seeds = label(seeds_bin)
        labels = watershed(-dist, seeds, mask=mask)
        return labels

    def loss_fn(self, output, target):
        loss_binary = self.loss_binary(output[:,0,:].unsqueeze(1), target["binary_mask"])
        loss_distance = self.loss_distance(output[:,1,:].unsqueeze(1), target["distance_map"])
        #loss_distance = self.loss_distance(output, target["distance_map"])

        loss = 2*(0.3 * loss_binary + 0.7 * loss_distance)
        #loss = loss_distance
        return loss

    def add_1x1_front(self, in_channels = 7):
        
        class NewModel(nn.Module):
            def __init__(self, model):
                super(NewModel, self).__init__()
                self.new_layer_1 = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1)
                self.new_layer_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
                self.model = model

            def forward(self, x):
                x = self.new_layer_1(x)  
                x = self.new_layer_2(x)
                x = self.model(x)     
                return x

        self.model = NewModel(self.model)

def prepare_input(img):
    H, W, C = img.shape
    pad_h = (32 - H % 32) if H % 32 != 0 else 0
    pad_w = (32 - W % 32) if W % 32 != 0 else 0
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    _max, _min = np.max(img, axis=(0,1)), np.min(img, axis=(0,1))
    img = (img - _min) / (_max - _min)
    img[np.isnan(img)] = 0.0  # If max == min... Shouldn't happen, but who knows at this point...
    img = torch.from_numpy(img).to(torch.float32)
    img = img.permute(2, 0, 1).unsqueeze(0) # B, C, H, W

    return img, pad_h, pad_w

def crop_to_original(img, pad_h, pad_w):
    if pad_h > 0:
        img = img[:-pad_h, :,]
    if pad_w > 0:
        img = img[:, :-pad_w,]
    return img
