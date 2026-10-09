import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from torch.utils.data import DataLoader
from data.Stage1_Dataset import Stage1Dataset

from modelling.Mesmer.MesmerNet import build_fresh, train_model
from misc.load_datasets import load_tissuenet, load_livecell

MODEL_TYPE = "Mesmer"  # Generic_Dataset.target_Mesmer builds (inner-distance, fgbg)
                        # targets straight from the instance label, same place every
                        # other model's target construction lives.
N_EPOCHS = 50
MODEL_NAME = "Mesmer_STAGE1_full"

print("--- CREATING DATALOADERS ---")

train_dataset_tn = load_tissuenet(MODEL_TYPE, split="train", transform=True)
test_dataset_tn = load_tissuenet(MODEL_TYPE, split="test", transform=False)
train_dataset_lc = load_livecell(MODEL_TYPE, split="train", transform=True)
test_dataset_lc = load_livecell(MODEL_TYPE, split="test", transform=False)

train_dataset = Stage1Dataset(train_dataset_tn, train_dataset_lc)
test_dataset = Stage1Dataset(test_dataset_lc, test_dataset_tn)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True)

print("--- TRAINING ---")

model = build_fresh()
train_model(
    model, train_loader, test_loader,
    model_name=MODEL_NAME,
    save_path=str(BENCH_DIR / "training/Mesmer/Stage_1/checkpoints"),
    epochs=N_EPOCHS,
    early_stopping_patience=5,
    save_every=5,
)
