# Following https://github.com/TIO-IKIM/CellViT/blob/main/cell_segmentation/datasets/pannuke.py
# and https://github.com/TIO-IKIM/CellViT/blob/main/cell_segmentation/utils/tools.py

import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt, label

def fix_duplicates(inst_map: np.ndarray) -> np.ndarray:
    """Re-label duplicated instances in an instance labelled mask.

    Parameters
    ----------
        inst_map : np.ndarray
            Instance labelled mask. Shape (H, W).

    Returns
    -------
        np.ndarray:
            The instance labelled mask without duplicated indices.
            Shape (H, W).
    """
    current_max_id = np.amax(inst_map)
    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    for inst_id in inst_list:
        inst = np.array(inst_map == inst_id, np.uint8)
        remapped_ids = label(inst)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        inst_map[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(inst_map)

    return inst_map

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

class Generator():
    @staticmethod
    def gen_distance_prob_maps(inst_map: np.ndarray) -> np.ndarray:
            """Generate distance probability maps
    
            Args:
                inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)
    
            Returns:
                np.ndarray: Distance probability map, shape (H, W)
            """
            inst_map = fix_duplicates(inst_map)
            dist = np.zeros_like(inst_map, dtype=np.float64)
            inst_list = list(np.unique(inst_map))
            if 0 in inst_list:
                inst_list.remove(0)
    
            for inst_id in inst_list:
                inst = np.array(inst_map == inst_id, np.uint8)
    
                y1, y2, x1, x2 = get_bounding_box(inst)
                y1 = y1 - 2 if y1 - 2 >= 0 else y1
                x1 = x1 - 2 if x1 - 2 >= 0 else x1
                x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
                y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2
    
                inst = inst[y1:y2, x1:x2]
    
                if inst.shape[0] < 2 or inst.shape[1] < 2:
                    continue
    
                # chessboard distance map generation
                # normalize distance to 0-1
                inst_dist = distance_transform_edt(inst)
                inst_dist = inst_dist.astype("float64")
    
                max_value = np.amax(inst_dist)
                if max_value <= 0:
                    continue
                inst_dist = inst_dist / (np.max(inst_dist) + 1e-10)
    
                dist_map_box = dist[y1:y2, x1:x2]
                dist_map_box[inst > 0] = inst_dist[inst > 0]
    
            return dist
    
    
    @staticmethod
    @njit
    def gen_stardist_maps(inst_map: np.ndarray) -> np.ndarray:
            """Generate StarDist map with 32 nrays
    
            Args:
                inst_map (np.ndarray): Instance-Map, each instance is has one integer starting by 1 (zero is background), Shape (H, W)
    
            Returns:
                np.ndarray: Stardist vector map, shape (n_rays, H, W)
            """
            n_rays = 32
            # inst_map = fix_duplicates(inst_map)
            dist = np.empty(inst_map.shape + (n_rays,), np.float32)
    
            st_rays = np.float32((2 * np.pi) / n_rays)
            for i in range(inst_map.shape[0]):
                for j in range(inst_map.shape[1]):
                    value = inst_map[i, j]
                    if value == 0:
                        dist[i, j] = 0
                    else:
                        for k in range(n_rays):
                            phi = np.float32(k * st_rays)
                            dy = np.cos(phi)
                            dx = np.sin(phi)
                            x, y = np.float32(0), np.float32(0)
                            while True:
                                x += dx
                                y += dy
                                ii = int(round(i + x))
                                jj = int(round(j + y))
                                if (
                                    ii < 0
                                    or ii >= inst_map.shape[0]
                                    or jj < 0
                                    or jj >= inst_map.shape[1]
                                    or value != inst_map[ii, jj]
                                ):
                                    # small correction as we overshoot the boundary
                                    t_corr = 1 - 0.5 / max(np.abs(dx), np.abs(dy))
                                    x -= t_corr * dx
                                    y -= t_corr * dy
                                    dst = np.sqrt(x**2 + y**2)
                                    dist[i, j, k] = dst
                                    break
    
            return dist.transpose(2, 0, 1)
