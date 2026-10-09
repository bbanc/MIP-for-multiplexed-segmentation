import os
import numpy as np
from data.Generic_Dataset import GenericDataset
import skimage


class CellposeDataset(GenericDataset):
    def __init__(self, image_paths, sample_idxs=None, transform=True, img_size=(256, 256), model_type=None, model_config=None,):
        # Todo: Check inputs
        super().__init__(transform, img_size, model_type, model_config)

        self.img_size = img_size
        self.MEANS = (7.18941135, 65.34027551,  0.)
        self.STDDEVS = (10.66316889, 37.83283596,  1.)
        self.image_paths = image_paths
        if sample_idxs is not None:
            self.image_paths = np.array(self.image_paths)[sample_idxs]

    def __len__(self):
        return len(self.image_paths)

    def from_idx(self, idx, transform, normalize, return_raw=False):
        """
        :param idx: data idx to load from
        :param transform: Whether to apply any transformations
        :param normalize: Whether to apply normalization
        :param return_raw: If true, returns the loaded image "as is" without target calculation or manipulations
        :return: image, labels and ID at idx
        """
        img_path = self.image_paths[idx]
        labels_path = img_path.replace("img", "masks")

        img, label = self.load_image(img_path, labels_path)

        if return_raw:
            img_name = os.path.basename(img_path)
            return img, label, img_name

        return self.generate_output(img, label, transform, normalize)
    
    def load_image(self, img_path, label_path):
        """
        :param img_path: path to image png
        :param label_path: path to labelled image
        :return: numpy array for image and labels (uint8/int16) respectively
        """
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            img = skimage.io.imread(img_path).astype(np.uint8)
            label = skimage.io.imread(label_path).astype(np.int16)
            
        return img, label
