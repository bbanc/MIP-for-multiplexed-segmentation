import numpy as np
import random

def preproc_img(img):
    """Takes H,W,C image, mean/std normalizes and returns as float with vals between 0 and 1"""
    #mean = np.mean(img, axis=(0, 1))
    #stddev = np.std(img, axis=(0, 1)) + np.finfo(float).eps  # Not 0
    #img = img - mean / stddev
    min_val, max_val = img.min(), img.max()
    print(min_val, max_val)
    n_img = ((img - min_val) / (max_val - min_val))

    return n_img.astype(np.float32)

def preproc_stack(stack):
    """Takes H,W,C image, mean/std normalizes and returns as float with vals between 0 and 1"""
    #mean = np.mean(img, axis=(0, 1))
    #stddev = np.std(img, axis=(0, 1)) + np.finfo(float).eps  # Not 0
    #img = img - mean / stddev
    min_val, max_val = np.min(stack, axis=(0,1)), np.max(stack, axis=(0,1))
    n_img = np.zeros_like(stack, dtype=np.float32)
    #n_img = ((stack - min_val) / (max_val - min_val)) # Need to handle 0
    for c in range(stack.shape[2]):  
        if max_val[c] > min_val[c]:  
            n_img[:, :, c] = (stack[:, :, c] - min_val[c]) / (max_val[c] - min_val[c])
        else:
            n_img[:, :, c] = 0.0  # or set to a constant value, e.g., 0.0 or 1.0
    
    return n_img.astype(np.float32)

def plot_examples(dataset, model, infer_img):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    random_idxs = [random.randint(0, len(dataset)) for _ in range(4)]
    for i, idx in enumerate(random_idxs):
        img, gt_label, _ = dataset.from_idx(idx, False, False, True)
        label = infer_img(img, model, gt_label)
        ax = show_result(img, label, ax=axs[i], linewidths=1.5)
        ax.set_axis_off()
    plt.show()
    return


def show_result(img, label, ax = None, scores = None, linewidths=1.5, colors=[ 'b', 'y', 'm', 'c', 'w'], linestyles=None):
    import matplotlib.pyplot as plt
    if not ax:
        fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_axis_off()

    unique_ints = np.unique(label)
    for i, idx in enumerate(unique_ints):
        if idx == 0:
            continue
        binary = label == idx
        ax.contour(binary, colors=colors[i % len(colors)], levels=[0.5], alpha=0.75, linewidths=linewidths, linestyles=linestyles)
    return ax