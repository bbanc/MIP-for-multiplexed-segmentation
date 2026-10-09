import numpy as np
import torch
import skimage
from scipy.ndimage import gaussian_filter


def post_process(masks):
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]

    filled_mask = np.zeros_like(masks)
    for label_value in unique_labels:
        filled_binary_mask = skimage.morphology.remove_small_holes(masks == label_value, max_size=20)
        filled_mask[filled_binary_mask] = label_value
    
    filled_mask = skimage.morphology.remove_small_objects(filled_mask, max_size=50)
    filled_mask = skimage.measure.label(filled_mask)
    return filled_mask

def prepare_input(img):
    H, W, C = img.shape
    pad_h = (32 - H % 32) if H % 32 != 0 else 0
    pad_w = (32 - W % 32) if W % 32 != 0 else 0
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    _max, _min = np.max(img, axis=(0,1)), np.min(img, axis=(0,1))
    img = (img - _min) / (_max - _min)
    img[np.isnan(img)] = 0.0  # If max == min... Shouldn't happen, but who knows at this point...
    img = torch.from_numpy(img).to(torch.float32)
    img = img.permute(2, 0, 1).unsqueeze(0) # B, C, H, W

    return img, pad_h, pad_w

def crop_to_original(img, pad_h, pad_w):
    if pad_h > 0:
        img = img[:-pad_h, :,]
    if pad_w > 0:
        img = img[:, :-pad_w,]
    return img

def infer_img(img_np, model, mask_thresh=0.6):
    img_input, pad_h, pad_w = prepare_input(img_np)
    model.eval()
    with torch.no_grad():
        out = model.model(img_input)
    
    binary_prob = out[:,0,:, :].sigmoid()
    dist_reg = out[:,1,:,:]
    
    mask = (binary_prob[0,] > mask_thresh).cpu().numpy()
    dist = (dist_reg[0,]).cpu().numpy()
    dist = crop_to_original(dist, pad_h, pad_w)
    
    dist_smooth = gaussian_filter(dist, sigma=2.0)
    
    mask = crop_to_original(mask, pad_h, pad_w)
    
    pad_width = 2  # Should be at least min_distance
    dist_padded = np.pad(dist_smooth, pad_width, mode='constant', constant_values=0)
    
    peak_coords = skimage.feature.peak_local_max(dist_padded, min_distance=10)
    peak_coords -= pad_width
    peak_coords = peak_coords[(peak_coords[:, 0] >= 0) & (peak_coords[:, 0] < dist.shape[0]) &
                              (peak_coords[:, 1] >= 0) & (peak_coords[:, 1] < dist.shape[1])]
    
    seeds_bin = np.zeros_like(mask, dtype=bool)
    seeds_bin[tuple(peak_coords.T)] = 1
    
    seeds = skimage.measure.label(seeds_bin)
    labels_ = skimage.segmentation.watershed(-dist_smooth, seeds, mask = mask)
    return post_process(labels_)

def infer_stack(stack, model, mask_thresh=0.6):
    return infer_img(stack, model, mask_thresh)
    

