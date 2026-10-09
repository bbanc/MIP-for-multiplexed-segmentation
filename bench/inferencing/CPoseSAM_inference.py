import os
import glob

import numpy as np
import skimage.io
import torch
from tqdm import tqdm

IMG_SIZE = 256

def _center_crop_or_pad(arr, size=IMG_SIZE):
    h, w = arr.shape[:2]
    out = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
    ch, cw = min(h, size), min(w, size)
    src_r0, src_c0 = (h - ch) // 2, (w - cw) // 2
    dst_r0, dst_c0 = (size - ch) // 2, (size - cw) // 2
    out[dst_r0:dst_r0 + ch, dst_c0:dst_c0 + cw, ...] = arr[src_r0:src_r0 + ch, src_c0:src_c0 + cw, ...]
    return out, (dst_r0, dst_c0, src_r0, src_c0, ch, cw)


def _normalize(stack):
    _max, _min = np.max(stack, axis=(0, 1)), np.min(stack, axis=(0, 1))
    n_img = np.zeros_like(stack, dtype=np.float32)
    for c in range(stack.shape[2]):
        if _max[c] > _min[c]:
            n_img[:, :, c] = (stack[:, :, c] - _min[c]) / (_max[c] - _min[c])
    return n_img


def infer_img(img, model, label=None, cellprob_threshold=0.0, flow_threshold=0.4):
    h, w = img.shape[:2]
    img = _normalize(img.astype(np.float32))
    cropped, (dst_r0, dst_c0, src_r0, src_c0, ch, cw) = _center_crop_or_pad(img)

    x = torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0).float().to(model.device)
    model.net.eval()
    with torch.no_grad():
        y, _ = model.net(x)
    y = y.cpu().numpy().transpose(0, 2, 3, 1)  # (1, 256, 256, 3)

    cellprob = y[..., 2]
    dP = y[..., :2].transpose((3, 0, 1, 2))

    cropped_label = model._compute_masks(
        [1, IMG_SIZE, IMG_SIZE, cropped.shape[2]], dP, cellprob,
        flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
        min_size=15, max_size_fraction=0.4, niter=200,
        stitch_threshold=0.0, do_3D=False,
    )

    label_out = np.zeros((h, w), dtype=cropped_label.dtype)
    label_out[src_r0:src_r0 + ch, src_c0:src_c0 + cw] = cropped_label[dst_r0:dst_r0 + ch, dst_c0:dst_c0 + cw]
    return label_out


def infer_stack(stack, model, label=None, cellprob_threshold=0.0, flow_threshold=0.4):
    return infer_img(stack, model, label=label,
                     cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)


def infer_img_zero_shot(img, model, cellprob_threshold=0.0, flow_threshold=0.4):
    masks, _, _ = model.eval(
        img, channel_axis=-1,
        cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold,
    )
    return masks


def infer_stack_zero_shot(stack, model, cellprob_threshold=0.0, flow_threshold=0.4):
    return infer_img_zero_shot(stack, model, cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)


def infer_dataset(dataset, model, save_dir):
    """
    :param dataset: Dataset class to infer
    :param model: CPSAM(-wrapped) model to use
    :param save_dir: Path to save
    :return: Writes labeled masks into save_path
    """
    if not os.path.exists(save_dir):
        print(save_dir, "does not exist, attempting to create...")
        os.makedirs(save_dir)
    else:
        labels = glob.glob(os.path.join(save_dir, "*.png"))
        if len(labels) == len(dataset):
            print("Already found matching inferred data... Skipping infer_dataset")
            return

    print("Beginning inference to", save_dir)
    for i in tqdm(range(len(dataset))):
        img, _, img_name = dataset.from_idx(i, False, False, return_raw=True)
        label = infer_img(img, model)
        if ".png" in str(img_name):
            skimage.io.imsave(os.path.join(save_dir, img_name), label, check_contrast=False)
        else:
            skimage.io.imsave(os.path.join(save_dir, f"{img_name}.png"), label, check_contrast=False)
