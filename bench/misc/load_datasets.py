from import_util import data_path
import os
import skimage.io

import glob
from data.CellposeDataset import CellposeDataset
from data.TissueNetDataset import TissueNetDataset
from data.LIVECellDataset import LIVECellDataset
from data.Stage1_Dataset import Stage1Dataset

def load_tissuenet(model_type, split="test", model_config=None, transform=False):
    assert model_type in ["CPose", "CPose2ch", "CPoseSAM", "CPN", "SAM", "DIST", "Stardist", "InstanSeg", "Mesmer", None]
    data_root = str(data_path("TissueNet"))
    dataset = None

    if split == "test":
        test_path = os.path.join(data_root, "tissuenet_v1.1_test.npz")
        dataset = TissueNetDataset(test_path, transform=transform, model_type=model_type, model_config=model_config)

    elif split == "train":
        train_path = os.path.join(data_root, "tissuenet_v1.1_train.npz")
        dataset = TissueNetDataset(train_path, transform=transform, model_type=model_type, model_config=model_config)

    elif split == "val":
        val_path = os.path.join(data_root, "tissuenet_v1.1_val.npz")
        dataset = TissueNetDataset(val_path, transform=transform, model_type=model_type, model_config=model_config)

    else:
        print("Make sure split is train/test/val")

    return dataset


def load_livecell(model_type, split="test", model_config=None, transform=False):
    assert model_type in ["CPose", "CPose2ch", "CPoseSAM", "CPN", "SAM", "DIST", "Stardist", "InstanSeg", "Mesmer", None]

    data_root = str(data_path("LIVECell/"))
    dataset = None

    if split == "test":
        imgs_path = os.path.join(data_root, "images", "livecell_test_images")
        json_path = os.path.join(data_root, "livecell_coco_test.json")
        dataset = LIVECellDataset(imgs_path, json_path, transform=transform, model_type=model_type, model_config=model_config)

    elif split == "train":
        imgs_path = os.path.join(data_root, "images", "livecell_train_val_images")
        json_path = os.path.join(data_root, "livecell_coco_train.json")
        dataset = LIVECellDataset(imgs_path, json_path, transform=transform, model_type=model_type, model_config=model_config)

    elif split == "val":
        imgs_path = os.path.join(data_root, "images", "livecell_train_val_images")
        json_path = os.path.join(data_root, "livecell_coco_val.json")
        dataset = LIVECellDataset(imgs_path, json_path, transform=transform, model_type=model_type, model_config=model_config)

    else:
        print("Make sure split is train/test/val")

    return dataset


def load_cposedata(model_type, sample_idxs=None, split="test", model_config=None, transform=False):
    assert model_type in ["CPose", "CPose2ch", "CPoseSAM", "CPN", "SAM", "Stardist", "DIST", "InstanSeg", "Mesmer", None]

    data_root = str(data_path("Cellpose"))
    dataset = None

    if split == "test":
        imgs_path = os.path.join(data_root, "test")
        img_pathes = glob.glob(os.path.join(imgs_path, "*_img.png"))
        dataset = CellposeDataset(img_pathes, sample_idxs=sample_idxs, transform=transform, model_type=model_type, model_config=model_config)

    elif split == "train":
        imgs_path = os.path.join(data_root, "train")
        img_pathes = glob.glob(os.path.join(imgs_path, "*_img.png"))
        dataset = CellposeDataset(img_pathes, sample_idxs=sample_idxs, transform=transform, model_type=model_type, model_config=model_config)

    elif split == "val":
        print("WARNING: Cellpose has no validation set, returning test set instead")
        imgs_path = os.path.join(data_root, "test")
        img_pathes = glob.glob(os.path.join(imgs_path, "*_img.png"))
        dataset = CellposeDataset(img_pathes, sample_idxs=sample_idxs, transform=transform, model_type=model_type, model_config=model_config)

    else:
        print("Make sure split is train/test/val/None")

    return dataset


def load_stage1_dataset(model_type, split="test", model_config=None, transform=False):
    assert model_type in ["CPose", "CPose2ch", "CPoseSAM", "CPN", "SAM", "DIST", "Stardist", "InstanSeg", "Mesmer", None]
    assert split in ["train", "test", "val"]
    
    tn_dataset = load_tissuenet(model_type, split, model_config, transform)
    lc_dataset = load_livecell(model_type, split, model_config, transform)

    dataset_stage1 = Stage1Dataset(tn_dataset, lc_dataset)
    return dataset_stage1


if __name__ == "__main__":
    dataset = load_tissuenet(None)
    img, _ = dataset.load_image(0)
    print(img.shape)
