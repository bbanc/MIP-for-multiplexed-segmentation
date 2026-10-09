import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.utils.data import DataLoader


from modelling.Cellpose.myCellpose import myCellpose
from misc.load_datasets import load_cposedata

MODEL_TYPE = "CPose2ch"
N_EPOCHS = 100
MODEL_NAME = "CPose2ch_STAGE2_full"
PRETRAINED_PATH = str(BENCH_DIR / "training/Cellpose/Stage_1/run/CPose2ch_STAGE1_full_BEST.pt")
LEARNING_RATE = 1e-4

print("--- CREATING DATALOADERS ---")

train_dataset = load_cposedata(MODEL_TYPE, split="train", transform=True)
test_dataset = load_cposedata(MODEL_TYPE, split="test", transform=False)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, )

print("--- TRAINING ---")
co = myCellpose(nchan=2, gpu=True, pretrained_model=PRETRAINED_PATH)

co._train_net(
    train_loader,
    test_loader=test_loader,
    early_stopping_patience=10,
    learning_rate=LEARNING_RATE,
    n_epochs=N_EPOCHS,
    model_name=MODEL_NAME,
    save_path=str(BENCH_DIR / "training/Cellpose/Stage_2/"),
    save_every=5,
)
