import sys
from pathlib import Path

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from torch.utils.data import DataLoader

from data.Stage1_Dataset import Stage1Dataset

from modelling.Cellpose.myCellpose import myCellpose
from misc.load_datasets import load_tissuenet, load_livecell

MODEL_TYPE = "CPose2ch"
N_EPOCHS = 50
MODEL_NAME = "CPose2ch_STAGE1_full"

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

print("--- TRAINING ---")
co = myCellpose(nchan=2, gpu=True)
co._train_net(
    train_loader,
    test_loader=test_loader,
    early_stopping_patience=5,
    n_epochs=N_EPOCHS,
    model_name=MODEL_NAME,
    save_path=str(BENCH_DIR / "training/Cellpose/Stage_1/run"),
    save_every=5,
)
