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
#from modelling.DistNet.DistNet import myCellpose
from modelling.UNet.DistNet import DIST_Net_Lightning
from misc.load_datasets import load_cposedata
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

MODEL_TYPE = "DIST"
N_EPOCHS = 100
MODEL_NAME = "DIST_STAGE2"

print("--- CREATING DATALOADERS ---")
ckpt_path = str(BENCH_DIR / "training/UNet/Stage_1/DIST_STAGE1_full_v2_final.ckpt")

train_dataset = load_cposedata(MODEL_TYPE, split="train", transform=True)
test_dataset = load_cposedata(MODEL_TYPE, split="test", transform=False)

batch_size = 16

#train_loader = IterationLoader(train_dataset, batch_size=batch_size, shuffle=True, max_iters=32, num_workers=8)
#test_loader = IterationLoader(test_dataset, batch_size=batch_size, shuffle=True, max_iters=32, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True)

print("--- TRAINING ---")
model = DIST_Net_Lightning.load_from_checkpoint(arch="unet", encoder_name="resnet50",
                                         in_channels=3, out_classes=2, 
                                         checkpoint_path=ckpt_path)

#model = DIST_Net_Lightning("unet", "resnet50", in_channels=3,
#                           out_classes=2, encoder_weights=None)

model.lr = 1e-4

checkpoint_callback = ModelCheckpoint(
    filename=MODEL_NAME + '_{epoch:02d}',
    save_top_k=-1,  
    every_n_epochs=5, 
)


trainer = pl.Trainer( 
    max_epochs=N_EPOCHS,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, early_stopping_callback],
    default_root_dir=str(BENCH_DIR / "training/UNet/Stage_2/lightning_logs" / MODEL_NAME),
)


trainer.fit(
    model, 
    train_dataloaders=train_loader, 
    val_dataloaders=test_loader,
)
trainer.save_checkpoint(str(BENCH_DIR / "training/UNet/Stage_2" / f"{MODEL_NAME}.ckpt"))
valid_metrics = trainer.validate(model, dataloaders=test_loader, verbose=False)
print(valid_metrics)
