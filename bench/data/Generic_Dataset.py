# This class is a generic dataset
import random

import skimage.morphology
import skimage.measure
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from skimage.segmentation import find_boundaries


class GenericDataset(Dataset):
    """Base dataset providing normalization, transforms, and target generation hooks."""
    def __getitem__(self, idx):
        """PyTorch entry point"""
        return self.from_idx(idx, self.transform, self.normalize)

    def __init__(self,  transform=True, img_size=(256,256), model_type = None, model_config=None,):
        super().__init__()
        self.transform = transform
        self.normalize = True
        self.img_size = img_size

        self.model_type = model_type
        self.model_config = model_config
        self.MEANS = None
        self.STDDEVS = None

    def __len__(self):
        raise NotImplementedError

    def from_idx(self, idx, transform, normalize, return_raw=False):
        raise NotImplementedError

    def generate_output(self, img, label, transform, normalize):
        """Convert raw image/label to tensors, normalize/augment, and build model-specific targets."""
        # To tensor
        img = torch.tensor(img).permute(2, 0, 1)  # (HWC to CHW)
        label = torch.tensor(label).unsqueeze(0)

        # Normalize
        if normalize:
            img = self._normalize(img)

        # Transform/resize
        if transform:
            img, label = self._transform(img, label)
            # print("Transform: ", img.shape, label.shape, img.dtype, label.dtype, img.min(), img.max())
        else:
            img = transforms.functional.center_crop(img, self.img_size)  # 0 pads if necessary
            label = transforms.functional.center_crop(label, self.img_size)
            # print("Cropping: ", img.shape, label.shape, img.dtype, label.dtype, img.min(), img.max())

        label = label.squeeze(0).numpy()
        label = skimage.morphology.label(label)
        label = torch.tensor(label).unsqueeze(0).to(torch.int32)
        img = torch.clip(img, min=0, max=1)
        # print(img.shape, label.shape)
        # Create model specific targets:

        if self.model_type == "CPose":
            return self.target_cpose(img, label)
        # print("Target: ", img.shape, label.shape, img.dtype, label.dtype)

        elif self.model_type == "CPose2ch":
            return self.target_cpose2ch(img, label)

        elif self.model_type == "CPoseSAM":
            return self.target_cpose(img, label)

        elif self.model_type == "SAM":
            #label mask must not be 0 for SAM (object sampler breaks), simply sample another if this happens
            if label.max() == 0: # Todo: CARE: THIS IS AN ENDLESS LOOP IF ALL MASK IMAGES ARE EMPTY (which should never happen)
                rng = np.random.default_rng(42)
                idx = rng.integers(0, len(self),)
                return self.__getitem__(idx)
            else:
                return img, label

        elif self.model_type == "Stardist":      
            return img, self.target_Stardist(label)

        elif self.model_type == "DIST":
            return img, self.target_DIST(label)

        elif self.model_type == "InstanSeg":
            return img, label

        elif self.model_type == "Mesmer":
            return self.target_Mesmer(img, label)

        else:
            if self.model_type != None:
                print("WARNING: unrecognized model_type, returning raw data,  ", self.model_type)
            return img, label


    def _normalize(self, img):
        """Normalize by channel statistics and rescale result to [0, 1]."""
        if not isinstance(img, torch.Tensor):
            img = torchvision.transforms.functional.pil_to_tensor(img).unsqueeze(0)

        n_img = transforms.functional.normalize(img.float(), self.MEANS, self.STDDEVS)
        min_val, max_val = n_img.min(), n_img.max()
        n_img = (n_img - min_val) / (max_val - min_val)
        return n_img

    def target_Stardist(self, label):
        """Generate Stardist distance/probability maps and supporting masks."""
        # Following: https://github.com/TIO-IKIM/CellViT/blob/main/cell_segmentation/datasets/pannuke.py
        masks = {}
        #print(label.shape, label.min(), label.max())
        from data.stardist_funcs import Generator
        dist_map = Generator.gen_distance_prob_maps(label.squeeze(0).numpy())
        stardist_map = Generator.gen_stardist_maps(label.squeeze(0).numpy())
        masks["dist_map"] = torch.Tensor(dist_map).type(torch.float32).unsqueeze(0)
        masks["stardist_map"] = torch.Tensor(stardist_map).type(torch.float32)
        masks["instance map"] = torch.Tensor(label).type(torch.int32)
        masks["binary_map"] = torch.Tensor(label > 0).type(torch.float32)
        return masks

    def target_cpose2ch(self, img, label):
        """Generate Cellpose binary/flow targets for two-channel inputs."""
        from cellpose import dynamics
        if img.shape[0] == 3:
            img = img[0:2,]
        label = label.squeeze().numpy()
        label = dynamics.labels_to_flows([label], device=torch.device("cpu"))[0]
        # Returns 4D: Labels, binary, X_flow, Y_Flow
        label = torch.tensor(label[1:, :, :])
        return img, label

    def target_DIST(self, label):
        """DIST targets: binary mask plus per-instance distance transform."""
        import scipy.ndimage
        ret = {}
        
        binary_mask = label.numpy()[0,:,:] > 0

        bounds = find_boundaries(label.numpy()[0,:,:], mode="outer")
        binary_mask = binary_mask * ~bounds        
        if not np.any(binary_mask):
            distance_map = np.zeros_like(binary_mask, dtype = np.float32)
            
        else:
            lab = label.numpy()[0,:,:]
            regs = skimage.measure.regionprops(lab)
            h, w = lab.shape
            distance_map = np.zeros((h, w), dtype = np.float32)

            for r in regs:
                minr, minc, maxr, maxc = r.bbox
                object_in_bbox = lab[minr:maxr, minc:maxc]
                binarized_object = (object_in_bbox == r.label)
                object_distance = scipy.ndimage.distance_transform_edt(binarized_object)
                object_distance = object_distance / object_distance.max()
                distance_map[minr:maxr, minc:maxc] += object_distance
                distance_map = np.clip(distance_map, 0, 1.0)
                    
        ret = {"binary_mask" : torch.from_numpy(binary_mask.astype(np.float32)).unsqueeze(0),
               "distance_map" : torch.from_numpy(distance_map.astype(np.float32)).unsqueeze(0),
                "labels" : label}
        
        return ret
    
    def target_Mesmer(self, img, label):
        from deepcell.image_generators import _transform_masks
        from tensorflow.keras.utils import to_categorical

        if img.shape[0] > 2:
            img = img[:2]  # (nuclear, membrane)

        lab = label.numpy().astype(np.int32)[0, :, :]  
        bounds = find_boundaries(lab, mode="outer") # As in DIST U-Net
        lab = lab * (~bounds) 
        label_np = lab[np.newaxis, ..., np.newaxis] 
        y_inner = _transform_masks(label_np, "inner-distance", data_format="channels_last",)[0]
        fgbg_binary = (lab > 0).astype(np.int32) 
        y_fgbg = to_categorical(fgbg_binary, num_classes=2).astype(np.float32)  
        targets = {
            "inner_distance": torch.tensor(y_inner, dtype=torch.float32).permute(2, 0, 1),
            "fgbg": torch.tensor(y_fgbg, dtype=torch.float32).permute(2, 0, 1),
        }
        return img, targets

    def target_cpose(self, img, label):
        """Cellpose targets for flows and segmentation"""
        from cellpose import dynamics
            
        label = label.squeeze().numpy()
        label = dynamics.labels_to_flows([label], device=torch.device("cpu"))[0]
        # Returns 4D: Labels, binary, X_flow, Y_Flow - ditch the labels
        label = torch.tensor(label[1:, :, :])
        return img, label
    
    def _transform(self, img, label,
                      prob_rota=0.5, prob_hflip=0.25, prob_vflip=0.25, prob_resize=0.25,
                      rota_max=180, ):

        """Random resized crop, rotation, and flips applied consistently to image/label."""
        # https://stackoverflow.com/questions/55261087/adding-gaussian-noise-to-image
        # Returns Tensors

        if not isinstance(img, torch.Tensor):
            img = transforms.functional.pil_to_tensor(img).unsqueeze(0)

        if not isinstance(label, torch.Tensor):
            label = transforms.functional.pil_to_tensor(label).unsqueeze(0)

        if prob_resize > random.uniform(0, 1):
            t, l, h, w = transforms.RandomResizedCrop(self.img_size).get_params(img, scale=[0.5, 1.5], ratio=[1, 1])

            img = transforms.functional.crop(img, t, l, h, w)
            img = transforms.functional.resize(img, self.img_size, antialias = True,
                                               interpolation=transforms.functional.InterpolationMode.BILINEAR)

            label = transforms.functional.crop(label, t, l, h, w)
            label = transforms.functional.resize(label, self.img_size, antialias = False,
                                                 interpolation=transforms.functional.InterpolationMode.NEAREST)
        else:
            img = transforms.functional.center_crop(img, self.img_size)  # 0 pads if necessary
            label = transforms.functional.center_crop(label, self.img_size)

        if prob_rota > random.uniform(0, 1):
            deg = random.uniform(0, rota_max)
            img = transforms.functional.rotate(img, deg, expand=False,
                                               interpolation=transforms.functional.InterpolationMode.BILINEAR)
            label = transforms.functional.rotate(label, deg, expand=False,
                                                 interpolation=transforms.functional.InterpolationMode.NEAREST)

        if prob_hflip > random.uniform(0, 1):
            img = transforms.functional.hflip(img)
            label = transforms.functional.hflip(label)

        if prob_vflip > random.uniform(0, 1):
            img = transforms.functional.vflip(img)
            label = transforms.functional.vflip(label)

        return img, label

    def calc_norm_vars(self):
        """Compute per-channel mean and stddev across the dataset."""
        mean = 0
        stddev = 0
        n_imgs = self.__len__()

        for i in range(n_imgs):
            img, _, _ = self.from_idx(i, False, False, return_raw=True)
            mean += np.mean(img, axis=(0, 1))
            stddev += np.std(img, axis=(0, 1))

        mean /= n_imgs
        stddev /= n_imgs

        return mean, stddev

    def load_data_to_np(self):
        """Load entire dataset into memory as numpy arrays and id strings."""
        imgs_list = []
        labels_list = []
        ids_list = []

        for idx in range(len(self)):
            img, label, _id = self.from_idx(idx, self.transform, self.normalize, return_raw=True)
            imgs_list.append(img)
            labels_list.append(label)
            ids_list.append(str(_id))

        return imgs_list, labels_list, ids_list
