import os
import numpy as np
import pandas as pd
from skimage import io
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from data.Stacked_Multiplexed import MultiplexedDataset
from validation.object_level import ObjectLevelMetrics, aggregate_sweep_rows
from validation.shape_level import ShapeLevelMetrics

SCORE_KEYS = ["Prec", "Rec", "F1", "mDICE", "mIoU", "PQ", "ID"]
SHAPE_KEYS = ["Area", "Perimeter", "Solidity", "Eccentricity", "Rectangularity",
              "Major Axis", "Minor Axis", "Compactness", "ID", "Aspect Ratios", "# Cells"]


def build_kfold_datasets(img_paths, label_paths, funcs, dapi_idxs, model_type, n_ch,
                          fold, n_splits_total=5, random_state=42):
    kf = KFold(n_splits=n_splits_total, shuffle=True, random_state=random_state)
    all_splits = list(kf.split(img_paths))
    if fold >= n_splits_total or fold < 0:
        raise ValueError(f"Invalid fold {fold}. Must be between 0 and {n_splits_total - 1}")

    train_idxs, val_idxs = all_splits[fold]
    train_dataset = MultiplexedDataset(img_paths[train_idxs], label_paths[train_idxs], dapi_idxs=dapi_idxs[train_idxs], img_stack_funcs=funcs[train_idxs], model_type=model_type)
    test_dataset = MultiplexedDataset(img_paths[val_idxs], label_paths[val_idxs], dapi_idxs=dapi_idxs[val_idxs], img_stack_funcs=funcs[val_idxs], model_type=model_type, transform=False)

    train_dataset.set_norm_vals(np.zeros(n_ch), np.ones(n_ch))
    test_dataset.set_norm_vals(np.zeros(n_ch), np.ones(n_ch))
    return train_dataset, test_dataset


def run_kfold_pipeline(img_paths, label_paths, funcs, dapi_idxs, model_type, n_ch,
                        fold, batch_size, train_fn, n_splits_total=5, random_state=42):
    train_dataset, test_dataset = build_kfold_datasets(
        img_paths, label_paths, funcs, dapi_idxs, model_type, n_ch,
        fold, n_splits_total=n_splits_total, random_state=random_state,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    return train_fn(train_loader, val_loader), test_dataset


def evaluate_model(model, dataset, csv_path, model_name, infer_fn):
    out_dir = os.path.join(csv_path, model_name)
    preds_dir = os.path.join(out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    scores, shapes, sweeps = [], [], []
    for i in range(len(dataset)):
        img, gt_label, _ = dataset.from_idx(i, False, False, return_raw=True)
        _id = str(dataset.label_pathes[i])
        pred_label = infer_fn(img, model).astype(np.uint32)

        olm = ObjectLevelMetrics(gt_label, pred_label, iou_thr=0.5, _id=_id)
        scores.append(olm.as_dict())
        sweeps.append(olm.threshold_stats())
        shapes.append({k: v[0] for k, v in ShapeLevelMetrics(pred_label, _id=_id).write_dict().items()})

        base, name = os.path.split(_id)
        name, _ = os.path.splitext(name)
        if name == "Annotation":
            name = os.path.basename(base)
        io.imsave(os.path.join(preds_dir, name + "_inferred.png"), pred_label, check_contrast=False)

    scores_df = pd.DataFrame(scores, columns=SCORE_KEYS)
    shapes_df = pd.DataFrame(shapes, columns=SHAPE_KEYS)
    scores_df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)
    shapes_df.to_csv(os.path.join(out_dir, "shapes.csv"), index=False)

    sweep_df = pd.DataFrame(sweeps)
    sweep_df.insert(0, "ID", scores_df["ID"])
    sweep_df.to_csv(os.path.join(out_dir, "sweep.csv"), index=False)
    aggregate_sweep_rows(sweeps).to_csv(os.path.join(out_dir, "sweep_summary.csv"), index=False)

    return scores_df, shapes_df
