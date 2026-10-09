import sys
from pathlib import Path

import numpy as np

BENCH_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "bench")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from deepcell_toolbox.deep_watershed import deep_watershed
from deepcell_toolbox.utils import tile_image, untile_image
from modelling.Mesmer.MesmerNet import IMG_SIZE


def infer_img(img_np, model, radius=1, maxima_threshold=0.05, interior_threshold=0.4):
    # Following: deepcell's application.py / deepcell_toolboy/utils.py
    h, w = img_np.shape[:2]
    nuc_mem = img_np[np.newaxis, :, :, :2].astype(np.float32) / 255.0  # (1,H,W,2)

    x_diff, y_diff = h - IMG_SIZE, w - IMG_SIZE
    if x_diff < 0 and y_diff < 0:
        x_diff, y_diff = abs(x_diff), abs(y_diff)
        x_pad = (x_diff // 2, x_diff - x_diff // 2)
        y_pad = (y_diff // 2, y_diff - y_diff // 2)
        padded = np.pad(nuc_mem, [(0, 0), x_pad, y_pad, (0, 0)], mode="constant")

        inner_pred, fgbg_pred = model.predict(padded, verbose=0)
        xl, yl = x_pad[0], y_pad[0]
        inner_pred = inner_pred[:, xl:xl + h, yl:yl + w, :]
        fgbg_pred = fgbg_pred[:, xl:xl + h, yl:yl + w, :]
    else:
        model_shape = (IMG_SIZE, IMG_SIZE)
        tiles, tiles_info = tile_image(nuc_mem, model_input_shape=model_shape, stride_ratio=0.75)
        inner_tiles, fgbg_tiles = model.predict(tiles, verbose=0)
        inner_pred = untile_image(inner_tiles, tiles_info, model_input_shape=model_shape)
        fgbg_pred = untile_image(fgbg_tiles, tiles_info, model_input_shape=model_shape)

    label = deep_watershed(
        [inner_pred, fgbg_pred[..., 1:2]],
        radius=radius, maxima_threshold=maxima_threshold, interior_threshold=interior_threshold,
    )[0, :, :, 0]
    return label


def infer_stack(stack, model, **kwargs):
    return infer_img(stack, model, **kwargs)
