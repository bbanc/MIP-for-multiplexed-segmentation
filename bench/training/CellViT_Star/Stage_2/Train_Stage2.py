import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import os

from torch.utils.data import DataLoader
import torch

from modelling.Cellvit.StarDistRN50_Rnd import StarDistRN50_Rnd
from misc.load_datasets import load_cposedata

MODEL_TYPE = "Stardist"
N_EPOCHS = 100
MODEL_NAME = "Stardist_STAGE2"

print("--- CREATING DATALOADERS ---")

ckpt_path = str(BENCH_DIR / "training/CellViT_Star/Stage_1/checkpoints/Stardist_STAGE1_full_BEST.pt")

train_dataset = load_cposedata(MODEL_TYPE, split="train", transform=True)
test_dataset = load_cposedata(MODEL_TYPE, split="test", transform=False)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

print("--- TRAINING ---")

model = StarDistRN50_Rnd(n_seg_cls=1, device="cuda:0")

model.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
model.load_model(ckpt_path)
model.run_training(
    train_loader,
    test_loader=test_loader,
    early_stopping_patience=10,
    model_name=MODEL_NAME,
    save_path=str(BENCH_DIR / "training/CellViT_Star/Stage_2/checkpoints"),
    save_every=5,
    n_epochs=N_EPOCHS,
)
