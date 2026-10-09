import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import os

from torch.utils.data import DataLoader
from data.Stage1_Dataset import Stage1Dataset

from modelling.Cellvit.StarDistRN50_Rnd import StarDistRN50_Rnd
from misc.load_datasets import load_tissuenet, load_livecell

MODEL_TYPE = "Stardist"
N_EPOCHS = 50
MODEL_NAME = "Stardist_STAGE1_full"

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

print("--- TRAINING ---")

model = StarDistRN50_Rnd(n_seg_cls=1, device="cuda:0")
model.run_training(train_loader, test_loader=test_loader, early_stopping_patience=5,
                   model_name=MODEL_NAME,
                   save_path=str(BENCH_DIR / "training/CellViT_Star/Stage_1/checkpoints"),
                   save_every=5,
                   n_epochs=N_EPOCHS)
