import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from torch.utils.data import DataLoader
from data.Stage1_Dataset import Stage1Dataset
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from modelling.UNet.DistNet import DIST_Net_Lightning
from misc.load_datasets import load_tissuenet, load_livecell
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

MODEL_TYPE = "DIST"
N_EPOCHS = 50
MODEL_NAME = "DIST_STAGE1_full_v2"

print("--- CREATING DATALOADERS ---")

train_dataset_tn = load_tissuenet(MODEL_TYPE, split="train", transform=True)
test_dataset_tn = load_tissuenet(MODEL_TYPE, split="test", transform=False)
train_dataset_lc = load_livecell(MODEL_TYPE, split="train", transform=True)
test_dataset_lc = load_livecell(MODEL_TYPE, split="test", transform=False)

train_dataset = Stage1Dataset(train_dataset_tn, train_dataset_lc)
test_dataset = Stage1Dataset(test_dataset_lc, test_dataset_tn)

batch_size = 16

#train_loader = IterationLoader(train_dataset, batch_size=batch_size, shuffle=True, max_iters=32, num_workers=8)
#test_loader = IterationLoader(test_dataset, batch_size=batch_size, shuffle=True, max_iters=32, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=5)

print("--- TRAINING ---")
model = DIST_Net_Lightning("unet", "resnet50", in_channels=3, out_classes=2, encoder_weights=None)

checkpoint_callback = ModelCheckpoint(
    filename= str(BENCH_DIR / "training/UNet/Stage_1/lightning_logs" / 'DIST_STAGE1_full_{epoch:02d}'),
    save_top_k=3, 
    every_n_epochs=5,  
    monitor="val_loss",
    mode = "min"
)

trainer = pl.Trainer( 
    max_epochs=N_EPOCHS,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, early_stopping_callback],
    default_root_dir=str(BENCH_DIR / "training/UNet/Stage_1/lightning_logs" / MODEL_NAME),
)

trainer.fit(
    model, 
    train_dataloaders=train_loader, 
    val_dataloaders=test_loader,
)

trainer.save_checkpoint(str(BENCH_DIR / "training/UNet/Stage_1" / f"{MODEL_NAME}_final.ckpt"))
valid_metrics = trainer.validate(model, dataloaders=test_loader, verbose=False)
print(valid_metrics)
