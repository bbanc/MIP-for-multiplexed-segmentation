import importlib.util
import sys
from pathlib import Path

# Since there's no pip install for Stardist with CellViT we're hacking into the library a bit here...

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from import_util import cellvit_path, CELLVIT_ROOT

if str(CELLVIT_ROOT) not in sys.path:
    sys.path.insert(0, str(CELLVIT_ROOT))

CELLVIT_CPP_PATH = cellvit_path("models/segmentation/cell_segmentation/cpp_net_stardist_rn50.py")

if not CELLVIT_CPP_PATH.exists():
    raise FileNotFoundError(f"CellViT cpp_net_stardist_rn50.py not found: {CELLVIT_CPP_PATH}")

spec = importlib.util.spec_from_file_location("cellvit_cpp_net_stardist_rn50", CELLVIT_CPP_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load CellViT module spec from {CELLVIT_CPP_PATH}")

cellvit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cellvit_module)

StarDistRN50 = cellvit_module.StarDistRN50
resnet50 = cellvit_module.resnet50


import torch
import os, datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class StarDistRN50_Rnd(StarDistRN50):
    def __init__(self, n_rays=32, n_seg_cls=6, device="cpu", lr=0.001):
        super(StarDistRN50_Rnd, self).__init__(n_rays, n_seg_cls)
        self.encoder = resnet50() #Want random init
        self.device = device
        self.learning_rate = lr
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.state = None
        self.encoder.to(self.device)
        self.to(self.device)


    def inference_step(self, img):
        """Input: Single numpy image (H,W,C) to run inference on (post preprocessing)"""

        img = torch.from_numpy(img).to(self.device)
        img = img.permute(2, 0, 1).unsqueeze(0) # B, C, H, W
        outputs = self(img)
        import torch.nn.functional as F
        
        outputs["binary_map"] = F.softmax(outputs["nuclei_type_map"], dim=1)  
        outputs["dist_map_sigmoid"] = F.sigmoid(outputs["dist_map"])
        
        (labels, _, _, ) = self.calculate_instance_map(outputs["dist_map_sigmoid"],
                                                        outputs["stardist_map"],
                                                        outputs["binary_map"])
        return labels[0,].cpu().numpy()


    def training_step(self, batch):
        imgs, gts = batch
        imgs = imgs.to(self.device)
        gts = self.gts_to_device(gts)
        self.train()
        self.optimizer.zero_grad()
        
        output = self(imgs)    
        loss = self.loss_fn(output, gts)
        loss.backward()
        self.optimizer.step()
        return loss

    def gts_to_device(self, gts):
        for key in gts:
            gts[key] = gts[key].to(self.device)
        return gts
 

    def validation_step(self, batch):
        imgs, gts = batch
        imgs = imgs.to(self.device)
        gts = self.gts_to_device(gts)
        with torch.no_grad():
            output = self(imgs)
            loss = self.loss_fn(output, gts)
        return loss

        
    def loss_fn(self, output, target):
        # Key: nuclei_type_map --> prob of pixel belonging to class = binary seg
        from base_ml.base_loss import MAEWeighted
        output["binary_map"] = output["nuclei_type_map"]
        #output["stardist_map"] = F.sigmoid(output["stardist_map"])
        LOSS_KEYS = ["dist_map", "stardist_map", "binary_map"]
        LOSS_FNS = [torch.nn.BCEWithLogitsLoss(), MAEWeighted(), torch.nn.BCEWithLogitsLoss()]
        
        total_loss = 0
        for criterion, key in zip(LOSS_FNS, LOSS_KEYS):
            # Apply the relevant loss
            if len(target[key].shape) == 3:
                target[key] = target[key].unsqueeze(0)
            total_loss += criterion(output[key], target[key])
        return total_loss        

    def run_epoch(self, data_loader, ):
        epoch_losses = []
        total_batches = len(data_loader)
        is_train = self.state == "train"
        self.train() if is_train else self.eval()
        for ibatch, batch in enumerate(data_loader):
            loss = self.training_step(batch) if is_train else self.validation_step(batch)
            epoch_losses.append(loss.item())
            self.tb_writer.add_scalar(f"Batch - Loss {self.state}", loss.item(), self.iepoch * total_batches + ibatch)
        self.train()
        return np.mean(epoch_losses)

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        
    def run_training(self, train_loader, test_loader=None, early_stopping_patience=25, save_path=None, save_every=100, n_epochs=10, model_name="Unnamed_dist_model"):

        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(save_path, f"{model_name}_{ts}") if save_path else None  # next to the checkpoints, as in myCellpose
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        best_val_loss = float('inf')
        best_file_name = None
        early_stop_counter = 0
        d = datetime.datetime.now()
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

        for iepoch in tqdm(range(n_epochs)):
            self.iepoch = iepoch
    
            self.tb_writer.flush()
            self.state = "train"
            train_loss = self.run_epoch(train_loader)

            self.tb_writer.add_scalar(f"Epoch - Loss {self.state}", train_loss, iepoch+1)
            if test_loader:
                self.state = "test"
                test_loss = self.run_epoch(test_loader)
                self.tb_writer.add_scalar(f"Epoch - Loss {self.state}", test_loss, iepoch+1)
                if test_loss < best_val_loss:
                    best_val_loss = test_loss
                    early_stop_counter = 0
                    file_name = "{}_BEST.pt".format(model_name)
                    file_name = os.path.join(save_path, file_name)
                    self.save_model(file_name)
                    best_file_name = file_name
               
                else: # TODO: Ugly
                    if early_stopping_patience > 0:
                        early_stop_counter += 1
                        if early_stop_counter >= early_stopping_patience:
                            print(f'Early stopping at {iepoch}, best model found at {iepoch - early_stopping_patience}')
                            break

    
            if save_path is not None:
                if iepoch+1 == n_epochs or (iepoch+1)%save_every == 0:
                    file_name = "{}_{}.pt".format(model_name, iepoch + 1)
                    file_name = os.path.join(save_path, file_name)
                    self.save_model(file_name)
            else:
                file_name = save_path
        
        self.tb_writer.close()

        if best_file_name is not None:
            self.load_model(best_file_name)
            print(f"Reloaded best checkpoint: {best_file_name}")

        return file_name

            
        
