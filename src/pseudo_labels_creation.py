import os
import numpy as np
import pandas
from tqdm import tqdm
import laspy

def update_attribute_where_cluster_match(coords_original_file_view, cluster, value):
    # Stack XYZ into a structured array for fast comparison
    # coords_A = np.stack((tile.x, tile.y, tile.z), axis=1)
    coords_B = np.stack((cluster.x, cluster.y, cluster.z), axis=1)

    # Use a hashable dtype to compare rows efficiently
    # coords_A_view = coords_A.view([('', coords_A.dtype)] * 3).reshape(-1)
    coords_B_view = coords_B.view([('', coords_B.dtype)] * 3).reshape(-1)

    # Find mask of matching indices in A
    mask = np.isin(coords_original_file_view, coords_B_view)

    return (mask, value)
    # print(np.sum(mask))
    tile.pseudo_label[mask] = value