import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from torch.utils.data import DataLoader

from modelling.Mesmer.MesmerNet import train_model, load_model
from misc.load_datasets import load_cposedata

MODEL_TYPE = "Mesmer"
N_EPOCHS = 100
MODEL_NAME = "Mesmer_STAGE2"

print("--- CREATING DATALOADERS ---")

ckpt_path = str(BENCH_DIR / "training/Mesmer/Stage_1/checkpoints/Mesmer_STAGE1_full_BEST.h5")

train_dataset = load_cposedata(MODEL_TYPE, split="train", transform=True)
test_dataset = load_cposedata(MODEL_TYPE, split="test", transform=False)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

print("--- TRAINING ---")

model = load_model(ckpt_path)

train_model(
    model, train_loader, test_loader,
    model_name=MODEL_NAME,
    save_path=str(BENCH_DIR / "training/Mesmer/Stage_2/checkpoints"),
    epochs=N_EPOCHS,
    early_stopping_patience=10,
    save_every=5,
)
