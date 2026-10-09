import numpy as np

from data.Generic_Dataset import GenericDataset
import skimage

import os
import glob


class MultiplexedDataset(GenericDataset):
    """Multiplexed Dataset:
    1) Read img_stack based on list of lists,
    2) Stacking func, 3) return"""
    
    def __init__(self, data_pathes, label_pathes, dapi_idxs=np.array([None]), img_stack_funcs=np.array([None]), transform=True, img_size=(256, 256), model_type=None, model_config=None, do_reshape=True):
        
        super().__init__(transform, img_size, model_type, model_config)
        
        self.data_pathes = data_pathes
        self.label_pathes = label_pathes
        self.stacking_funcs = img_stack_funcs
        self.dapi_idxs = dapi_idxs
        self.n_channels = None
        #self.stack_indices = stack_indices
        self.do_reshape = do_reshape
        self.get_norm_vals()
        
    def __len__(self):
        return len(self.data_pathes)

    def from_idx(self, idx, transform, normalize, return_raw=False):
        img, label = self.load_image(idx)

        if return_raw:
            return img, label, str(idx)  # Makes it a bit easier to see where the data came from

        return self.generate_output(img, label, transform, normalize)

    def load_image(self, idx):
    # Load label and image
        label = skimage.io.imread(self.label_pathes[idx])
        img_stack = skimage.io.imread(self.data_pathes[idx]) 

        #print("load_image:", img_stack.shape)
        if len(img_stack.shape) > 3:
            N, H, W, C = img_stack.shape
            img_stack = np.moveaxis(img_stack, 0, -1).reshape(H, W, N * C,)

        elif (img_stack.shape[0] < img_stack.shape[1]) and (img_stack.shape[0] < img_stack.shape[2]):  # (C, H, W) format, Vectra
            img_stack = np.transpose(img_stack, (1, 2, 0))

        # DAPI channel index
        dapi_idx = int(self.dapi_idxs[idx]) if self.dapi_idxs[idx] is not None else None
        #print("load_image:", dapi_idx)

        # Min-max normalization per channel
        _max = np.max(img_stack, axis=(0,1))
        _min = np.min(img_stack, axis=(0,1))
        #print("load_image:", _max, _min)
        range_val = np.where((_max - _min) == 0, 1, _max - _min)
        img_stack = (((img_stack - _min) / range_val) * 255).astype(np.uint8)
        
        # Store number of channels
        self.n_channels = img_stack.shape[-1]
        #print("load_image:", img_stack.shape[-1])

        # Swap DAPI channel to first position
        if dapi_idx is not None and 0 <= dapi_idx < self.n_channels:
            channels = [dapi_idx] + [i for i in range(img_stack.shape[-1]) if i != dapi_idx]
            #print("Trying to reshape...")
            img_stack = img_stack[:,:,channels]
        
        #print(img_stack.shape)
        if self.stacking_funcs is not None and len(self.stacking_funcs) > 0:
            stacking_func = self.stacking_funcs[idx]
            img_stack = stacking_func(img_stack)

        #print(img_stack.shape)
        
        return img_stack, label.astype(np.int16)

    
    def get_n_channels(self):
        if self.n_channels:
            return self.n_channels

        # Todo: 
        img = skimage.io.imread(self.data_pathes[0])
        if len(img.shape) == 4:
            img = img.transpose(0, 3, 1, 2).reshape(-1, img.shape[1], img.shape[2]) # Reshapes NHWC to (N*C)HW
        self.n_channels = img.shape[0]
        return self.n_channels

    def get_norm_vals(self):
        if not self.n_channels:
            self.get_n_channels()
        self.MEANS = np.zeros(self.n_channels)
        self.STDDEVS = np.ones(self.n_channels)

    def set_norm_vals(self, means, stddevs):
        self.MEANS = means
        self.STDDEVS = stddevs


def img_stackg_f(img_stack):
    return img_stack
    
def max_project(stack):
    """Reduce a DAPI-first HWC stack to 3 channels: (DAPI, max-of-rest, unused)."""
    h, w, c = stack.shape
    out = np.zeros((h, w, 3), dtype=stack.dtype)
    out[:, :, 0] = stack[:, :, 0]
    out[:, :, 1] = np.max(stack[:, :, 1:], axis=-1)
    return out

def img_stack_max(img_stack):
    # IN: CHW, out HWC
    _max = np.max(img_stack, axis=(0,1))
    _min = np.min(img_stack, axis=(0,1))
    range_val = np.where((_max - _min) == 0, 1, _max - _min)
    img_stack = (((img_stack - _min) / range_val) * 255).astype(np.uint8) # Min-Max Rescale to uint8
    return max_project(img_stack)

def zip2img(zip_p, img_shape):
    from roifile import ImagejRoi
    rois = ImagejRoi.fromfile(zip_p)
    label_img = np.zeros(img_shape, dtype=np.int16)
    for i, r in enumerate(rois):
        coords = r.coordinates() # (N,2) --> X, Y order, need Y, X
        poly = skimage.draw.polygon(coords[:,1], coords[:,0], shape=img_shape)
        label_img[poly] = i
    return label_img

    
def write_labels_to_folder(folder_p):
    print("Processing: ", folder_p)
    zip_p = glob.glob(os.path.join(folder_p, "*.zip"))[0]
    img_p = glob.glob(os.path.join(folder_p, "*-Crop_Tif.tif"))
    img_p.sort()
    img_p = img_p[-1]
    img = skimage.io.imread(img_p)
    h = img.shape[1]
    w = img.shape[2]
    
    img_n = os.path.basename(img_p)
    img_n, _ = os.path.splitext(img_n)
    label_img = zip2img(zip_p, (h,w))    
    skimage.io.imsave(os.path.join(folder_p, img_n + "_LABELS.png"), label_img, check_contrast=False)


def write_all_labels():
    import sys
    from pathlib import Path

    BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
    if str(BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(BENCH_DIR))

    from import_util import data_path

    p = str(data_path("Multiplex/"))
    modalities = ["CODEX", "Zeiss", "Vectra"]
    for mod in modalities:
        if mod == "Vectra":
            experiment_pathes = glob.glob(os.path.join(p, mod, "*/"))
            for exp_p in experiment_pathes:
                folder_pathes = glob.glob(os.path.join(p, mod, exp_p, "*/"))
        
                for folder_p in folder_pathes:
                    write_labels_to_folder(folder_p)
        else:
            folder_pathes = glob.glob(os.path.join(p, mod, "*/"))
            
            for folder_p in folder_pathes:
                write_labels_to_folder(folder_p)
        
