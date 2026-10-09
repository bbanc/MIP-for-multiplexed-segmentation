import os
import numpy as np

from pycocotools.coco import COCO
from data.Generic_Dataset import GenericDataset
import skimage

class LIVECellDataset(GenericDataset):
    
    def __init__(self, data_dir, json_path, transform=True, img_size=(256, 256), model_type=None,
                 model_config=None,):
        super().__init__(transform, img_size, model_type, model_config)
        self.data_dir = data_dir
        self.MEANS = (0., 10.56812764,  0.)
        self.STDDEVS = (1., 18.88155906,  1.)

        self.solution = COCO(json_path)
        # self.solution_ids = self.solution.getImgIds()  # CARE: The original LIVECell contains duplicates...
        # This breaks the validation, so we need to filter out non unique file_names:
        # Is there a more efficient way to do this?

        test = set()
        ids_to_keep = []
        for _id, img_entry in self.solution.imgs.items():
            if img_entry["file_name"] in test:
                continue
            else:
                test.add(img_entry["file_name"])
                ids_to_keep.append(_id)

        # self.solution_ids = {k: v for k, v in self.solution.imgs.items() if k not in ids_to_delete}
        self.solution_ids = ids_to_keep

    def __len__(self):
        return len(self.solution_ids)

    def from_idx(self, idx, transform, normalize, return_raw=False):
        _id = self.solution_ids[idx]
        img, label = self.load_image(_id)
        name = self.solution.imgs[_id]["file_name"]
        if return_raw:
            return img, label, name
        return self.generate_output(img, label, transform, normalize)
    
    def load_image(self, _id):
        """Returns numpy"""
        fp = os.path.join(self.data_dir, self.solution.imgs[_id]["file_name"])
        img = skimage.io.imread(fp)
        # The images are different than "normal" images: Median 128, Positive + Negative signal, Very noisy.
        # This takes care of all of that and makes them look and behave a bit more similar to fluorescence images      

        n_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=float)
        n_img[:, :, 1] = np.abs((img - np.mean(img)) / np.std(img))
        n_img[:, :, 1] = n_img[:, :, 1] / n_img[:, :, 1].max()
        n_img = (n_img * (2**8 - 1)).astype(np.uint8)

        labels = np.zeros(img.shape, dtype=np.int16)
        ann_ids = self.solution.getAnnIds(imgIds=[_id])
        anns = self.solution.loadAnns(ann_ids)
        i = 1
        for ann in anns:
            label = self.solution.annToMask(ann)
            labels[label>0] = i
            i+=1

        return n_img, labels

