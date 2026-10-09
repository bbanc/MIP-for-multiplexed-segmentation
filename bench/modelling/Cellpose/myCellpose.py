import os
import datetime
os.environ["TQDM_DISABLE"] = "STOP"
import numpy as np
import torch
from torch import nn
TORCH_ENABLED = True
from torch.utils.tensorboard import SummaryWriter
from cellpose import models
from tqdm import tqdm

class myCellpose(models.CellposeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _train_net(self, train_loader,
                new_train=False, test_loader=None, test_labels=None, early_stopping_patience=25,
                save_path=None, save_every=100, save_each=False,
                n_epochs=10, learning_rate=0.01, momentum=0.9, weight_decay=0.001,
                SGD=False, batch_size=8, nimg_per_epoch=None, rescale=True, model_name=None):
        
        """Main train function - adapted from main implementation"""

        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_path:
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{model_name}_{ts}" if model_name else f"run_{ts}"
            tb_log_dir = os.path.join(save_path, run_name)
            os.makedirs(tb_log_dir, exist_ok=True)
            TB_writer = SummaryWriter(log_dir=tb_log_dir)
        else:
            TB_writer = SummaryWriter()
        
        self.n_epochs = n_epochs 
        self.optimizer = torch.optim.AdamW(self.net.parameters(),
                                     lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)

        self.learning_rate_const = learning_rate
        self.learning_rate = learning_rate        

        self.train_losses = []
        self.test_losses = []

        best_val_loss = float('inf')
        best_file_name = None

        d = datetime.datetime.now()

        # cannot train with mkldnn
        self.net.mkldnn = False      
        
        # ---MAIN LOOP---
        
        for iepoch in tqdm(range(self.n_epochs)):    
            self.train_losses = []
            for ibatch, batch in enumerate(train_loader):       
                imgi, lbl = batch

                train_l = self._train_step(imgi, lbl)
                self.train_losses.append(train_l.item())
                
                TB_writer.add_scalar("Batch - Loss train", train_l.item(), (iepoch + 1) * (ibatch+1))

                #print(train_loss)

            TB_writer.add_scalar("Epoch - Loss train", np.mean(self.train_losses), iepoch+1)  
            imgs = []
            #print(imgi.shape, lbl.shape)

            #for i in range(imgi.shape[0]):
            #    imgs.append((((imgi[i,:,:,0] - imgi[i,:,:,0].min()) / imgi[i,:,:,0].max()) * 255).to(torch.uint8))
            #img_grid = make_grid(imgs)
            #TB_writer.add_image("Example_Grid_Train", img_grid, global_step = iepoch)

            if save_path is not None:
                if iepoch+1 == self.n_epochs or (iepoch+1)%save_every == 0:
                    file_name = "{}_{}.pt".format(model_name, iepoch + 1)
                    file_name = os.path.join(save_path, file_name)
                    self.net.save_model(file_name)
            else:
                file_name = save_path
                
            if test_loader:
                self.test_losses = []
                for ibatch, batch in enumerate(test_loader):       
                    imgi, lbl = batch
                    test_l = self._val_step(imgi, lbl)
                    self.test_losses.append(test_l.item())   
                    TB_writer.add_scalar("Batch - Loss test", test_l.item(), (iepoch + 1) * (ibatch+1))

                imgs = []
                #for i in range(imgi.shape[0]):
                #    imgs.append((((imgi[i,:,:,0] - imgi[i,:,:,0].min()) / imgi[i,:,:,0].max()) * 255).to(torch.uint8))
                #img_grid = make_grid(imgs)
                #TB_writer.add_image("Example_Grid_Test", img_grid, global_step = iepoch)

                mean_test_loss = np.mean(self.test_losses)
                TB_writer.add_scalar("Epoch - Loss test", mean_test_loss, iepoch+1)

                if mean_test_loss < best_val_loss:
                    best_val_loss = mean_test_loss
                    early_stop_counter = 0
                    file_name = "{}_BEST.pt".format(model_name)
                    file_name = os.path.join(save_path, file_name)
                    self.net.save_model(file_name)
                    best_file_name = file_name
            
                else: # TODO: Ugly
                    if early_stopping_patience > 0:
                        early_stop_counter += 1
                        if early_stop_counter >= early_stopping_patience:
                            print(f'Early stopping at {iepoch}, best model found at {iepoch - early_stopping_patience}')
                            break
                            
            TB_writer.flush()

        # reset to mkldnn if available
        self.net.mkldnn = self.mkldnn
        TB_writer.close()

        if best_file_name is not None:
            self.net.load_model(best_file_name, device=self.device)
            print(f"Reloaded best checkpoint: {best_file_name}")

        return file_name
    
    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        criterion = nn.MSELoss(reduction="mean")
        criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
        veci = (5. * lbl[:,1:]).to(self.device)
        lbl = (lbl[:,0]>.5).float().to(self.device)
        loss = criterion(y[:,:2] , veci) 
        loss /= 2.        
        loss2 = criterion2(y[:,2] , lbl)
        loss = loss + loss2
        return loss   

    def _train_step(self, x, lbl):
        #print(x.shape, x.dtype, x.max(), "  ", lbl.shape, lbl.dtype, lbl.max())
        X = x.float().to(self.device)
        self.optimizer.zero_grad()
        self.net.train()
        y = self.net(X)[0]
        del X  
        loss = self.loss_fn(lbl, y)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def _val_step(self, x, lbl):
        X = x.float().to(self.device)
        self.net.eval()
        with torch.no_grad():
            y = self.net(X)[0]
            del X
            loss = self.loss_fn(lbl,y)
        return loss
        

def add1x1front(cp_model, in_ch=3):
    import torch.nn as nn
    new_layer = nn.Conv2d(in_ch, 2, kernel_size=1)
    new_layer2 = nn.Conv2d(2, 2, kernel_size=1)

    res_down = cp_model.net.downsample
    res_down = nn.Sequential(new_layer, new_layer2, res_down)
    
    cp_model.net.downsample = res_down
    cp_model.net.to(cp_model.device)
    return cp_model
