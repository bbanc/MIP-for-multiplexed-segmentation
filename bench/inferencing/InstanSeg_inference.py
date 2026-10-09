import numpy as np
import torch


def prepare_input(img):
    H, W, C = img.shape
    pad_h = (32 - H % 32) if H % 32 != 0 else 0
    pad_w = (32 - W % 32) if W % 32 != 0 else 0
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    _max, _min = np.max(img, axis=(0, 1)), np.min(img, axis=(0, 1))
    img = (img - _min) / np.where(_max - _min == 0, 1, _max - _min)
    img = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img, pad_h, pad_w


def crop_to_original(label, pad_h, pad_w):
    if pad_h > 0:
        label = label[:-pad_h, :]
    if pad_w > 0:
        label = label[:, :-pad_w]
    return label


def infer_img(img_np, model):
    device = next(model.backbone.parameters()).device
    x, pad_h, pad_w = prepare_input(img_np)
    model.backbone.eval()
    with torch.no_grad():
        prediction = model.backbone(x.to(device))
        label = model.method.postprocessing(prediction[0])  # (1, H, W) -- leading axis is instance-type (nuclei/cells); size 1 here since cells_and_nuclei=False
    label = label[0].cpu().numpy()  # -> (H, W)
    return crop_to_original(label, pad_h, pad_w)


def infer_stack(stack, model):
    return infer_img(stack, model)
