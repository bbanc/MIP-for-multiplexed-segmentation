import numpy as np
from data.Generic_Dataset import GenericDataset


class TissueNetDataset(GenericDataset):
    """TissueNet data is saved in a single npz file that gets read to disk"""
    def __init__(self, data_path, transform=True, img_size=(256, 256), model_type=None, model_config=None):
        super().__init__(transform, img_size, model_type, model_config)
        self.MEANS = (26.33647236, 23.95146017,  0.)
        self.STDDEVS = (28.78034246, 28.17420131,  1.)

        load = np.load(data_path)
        self.imgs_data, self.label_data = load["X"].copy(), load["y"].copy()
        load.close()      
        
    def __len__(self):
        return self.imgs_data.shape[0]

    def from_idx(self, idx, transform, normalize, return_raw=False):
        img, label = self.load_image(idx)

        if return_raw:
            return img, label, str(idx) + ".png"  # Makes it a bit easier

        return self.generate_output(img, label, transform, normalize)

    def load_image(self, idx):
        img = self.imgs_data[idx, :]
        label = self.label_data[idx,:,:,0] # Membrane seg
            
        # Theres probably a more efficient way to do this 
        # Take (H,W,2) with Tissuenets relative intensity scale, rescale into standard (H, W, 3) uint8
        # In TN - membranes: ch1, nuclei: ch0
        # We want RGB - Red: nucleus, Green: Membrane, Blue: ...
        
        n_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)   
        # ???  
        n_img[:,:,0] = ((img[:,:,0] - img[:,:,0].min()) / (img[:,:,0].max() - img[:,:,0].min())) * 255 # Min-Max Rescale on channel 0
        n_img[:,:,1] = ((img[:,:,1] - img[:,:,1].min()) / (img[:,:,1].max() - img[:,:,1].min())) * 255 # Min-Max Rescale on channel 1
                    
        return n_img, label.astype(np.int16)

