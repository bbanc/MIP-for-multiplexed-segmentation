import os
import skimage.io
from tqdm import tqdm
import glob


from inferencing.utils import preproc_img, preproc_stack

def infer_img(img, model, label=None, cellprob_threshold=0.4, flow_threshold=0.8):
    """
    Runs inference using a loaded cellpose model and an image
    :param img: Input image (uint8)
    :param model: Cellpose model
    :param cellprob_threshold: Cellpose thresh on binary segmentation (Higher = fewer pixels)
    :param flow_threshold: Cellpose thresh on flow estimation (Higher = more pixels)
    :return: labeled image of the inferred image
    """
    img = preproc_img(img)
    masks, flows, _ = model.eval(img, normalize=False, channels=None,
                                      cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)
    return masks


def infer_stack(stack, model, label=None, cellprob_threshold=0.4, flow_threshold=0.8):
    import torch
    stack = preproc_stack(stack)
    imgs = torch.from_numpy(stack).permute(2,0,1).unsqueeze(0).float().to("cpu")
    with torch.no_grad():
        y, style = model.net(imgs)[:2]
    y = y.numpy()
    y = y.transpose(0,2,3,1)
    style = style.numpy()

    cellprob = y[..., 2]
    dP = y[..., :2].transpose((3, 0, 1, 2))
    style = style.squeeze()

    h, w, c = stack.shape
    
    masks = model._compute_masks([1, h, w, c], dP, cellprob, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, interp=True, min_size=15, max_size_fraction=0.4, niter=200,
                stitch_threshold=0.0, do_3D=False)
    
    return masks


def infer_dataset(dataset, cp_model, save_dir):
    """
    :param dataset: Dataset class to infer
    :param cp_model: Cellpose model to use
    :param save_dir: Path to save
    :return: Writes labeled masks into save_path
    """

    if not os.path.exists(save_dir):
        print(save_dir,
              "does not exist, attempting to create...")
        os.makedirs(save_dir)
    else:
        # Check content
        labels = glob.glob(os.path.join(save_dir, "*.png"))
        if len(labels) == len(dataset):
            print("Already found matching inferred data... Skipping infer_dataset")
            return

    print("Beginning inference to", save_dir)
    for i in tqdm(range(len(dataset))):
        img, _, img_name = dataset.from_idx(i, False, False, return_raw=True)
        label = infer_img(img, cp_model)
        if ".png" in str(img_name):
            skimage.io.imsave(os.path.join(save_dir, img_name), label, check_contrast=False)
        else:
            skimage.io.imsave(os.path.join(save_dir, f"{img_name}.png"), label, check_contrast=False)

