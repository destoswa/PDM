import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
import laspy
import json
import scipy
import copy
import pickle
from tqdm import tqdm
from scipy.spatial import cKDTree


def remove_duplicates(laz_file):
    # Find pairs of points
    coords = np.round(np.vstack((laz_file.x, laz_file.y, laz_file.z)),2).T
    tree_B = cKDTree(coords)
    pairs = tree_B.query_pairs(1e-2)

    # Create the mask with dupplicates
    mask = [True for i in range(len(coords))]
    for pair in pairs:
        mask[pair[1]] = False

    # Remove the dupplicates from the file
    laz_file.points = laz_file.points[mask]

def remove_duplicates_v2(laz_file):
    # Round coordinates to 2 decimals and stack into Nx3 array
    coords = np.round(np.vstack((laz_file.x, laz_file.y, laz_file.z)).T, 2)

    # Build KDTree and find pairs of close points (within 1 cm)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(1e-2)  # 1 cm tolerance

    # Create boolean mask: keep first occurrence, drop the second
    mask = np.ones(len(coords), dtype=bool)
    for i, j in pairs:
        mask[j] = False

    # Create new LAS object
    header = laspy.LasHeader(point_format=laz_file.header.point_format, version=laz_file.header.version)
    header.point_count = 0
    new_las = laspy.LasData(header)

    # Copy all dimensions with the mask applied
    for dim in laz_file.point_format.dimension_names:
        setattr(new_las, dim, getattr(laz_file, dim)[mask])

    return new_las

def flattening_tile(tile_src, grid_size=10, verbose=True):
    # load file
    laz = laspy.read(tile_src)
    init_len = len(laz)
    laz = remove_duplicates_v2(laz)
    if verbose:
        print(f"Removing duplicates: From {init_len} to {len(laz)}")
    laz.write(tile_src)
    points = np.vstack((laz.x, laz.y, laz.z)).T
    points_flatten = copy.deepcopy(points)
    points_interpolated = copy.deepcopy(points)

    # Divide into tiles and find local minimums
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    x_bins = np.append(np.arange(x_min, x_max, grid_size), x_max)
    y_bins = np.append(np.arange(y_min, y_max, grid_size), y_max)

    grid = {i:{j:[] for j in range(y_bins.size - 1)} for i in range(x_bins.size -1)}
    for _, (px, py, pz) in tqdm(enumerate(points), total=len(points), desc="Creating grid", disable=verbose==False):
        xbin = np.clip(0, (px - x_min) // grid_size, x_bins.size - 1)
        ybin = np.clip(0, (py - y_min) // grid_size, y_bins.size - 1)
        grid[xbin][ybin].append((px, py, pz))

    # Create grid_min
    grid_used = np.zeros((x_bins.size - 1, y_bins.size - 1))
    lst_grid_min = []
    lst_grid_min_pos = []
    for x in grid.keys():
        for y in grid[x].keys():
            if np.array(grid[x][y]).shape[0] > 0:
                grid_used[x, y] = 1
                lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
                arg_min = np.argmin(np.array(grid[x][y])[:,2])
                lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2])
            else:
                grid_used[x, y] = 0
    arr_grid_min_pos = np.vstack(lst_grid_min_pos)
    if verbose:
        print("Resulting grid:")
        print(arr_grid_min_pos.shape)
        print(grid_used)

    # Interpolate
    points_xy = np.array(points)[:,0:2]
    interpolated_min_z = scipy.interpolate.griddata(arr_grid_min_pos, np.array(lst_grid_min), points_xy, method="cubic", fill_value=-1)

    mask_valid = np.array([x != -1 for x in list(interpolated_min_z)])
    points_interpolated = points_interpolated[mask_valid]
    points_interpolated[:, 2] = interpolated_min_z[mask_valid]

    if verbose:
        print("Interpolation:")
        print(f"Original number of points: {points.shape[0]}")
        print(f"Interpollated number of points: {points_interpolated.shape[0]} ({int(points_interpolated.shape[0] / points.shape[0]*100)}%)")

    # save floor
    filtered_points = {dim: getattr(laz, dim)[mask_valid] for dim in laz.point_format.dimension_names}
    header = laspy.LasHeader(point_format=laz.header.point_format, version=laz.header.version)
    new_las = laspy.LasData(header)

    #   _Assign filtered and modified data
    for dim, values in filtered_points.items():
        setattr(new_las, dim, values)
    # new_las.xyz = points_interpolated
    setattr(new_las, 'x', points_interpolated[:,0])
    setattr(new_las, 'y', points_interpolated[:,1])
    setattr(new_las, 'z', points_interpolated[:,2])

    #   _Save new file
    floor_dir = os.path.join(os.path.dirname(tile_src), 'floor')
    os.makedirs(floor_dir, exist_ok=True)
    new_las.write(os.path.join(floor_dir, os.path.basename(tile_src).split('.laz')[0] + f"_floor_{grid_size}m.laz"))
    if verbose:
        print("Saved file: ", os.path.join(floor_dir, os.path.basename(tile_src).split('.laz')[0] + f"_floor_{grid_size}m.laz"))

    # Flatten
    points_flatten = points_flatten[mask_valid]
    points_flatten[:,2] = points_flatten[:,2] - points_interpolated[:,2]

    filtered_points = {dim: getattr(laz, dim)[mask_valid] for dim in laz.point_format.dimension_names}
    header = laspy.LasHeader(point_format=laz.header.point_format, version=laz.header.version)
    header.point_count = 0
    new_las = laspy.LasData(header)


    #   _Assign filtered and modified data
    for dim, values in filtered_points.items():
        setattr(new_las, dim, values)

    setattr(new_las, 'x', points_flatten[:,0])
    setattr(new_las, 'y', points_flatten[:,1])
    setattr(new_las, 'z', points_flatten[:,2])

    #   _Save new file
    flatten_dir = os.path.join(os.path.dirname(tile_src), 'flatten')
    os.makedirs(flatten_dir, exist_ok=True)
    new_las.write(os.path.join(flatten_dir, os.path.basename(tile_src).split('.laz')[0] + f"_flatten_{grid_size}m.laz"))
    if verbose:
        print("Saved file: ", os.path.join(flatten_dir, os.path.basename(tile_src).split('.laz')[0] + f"_flatten_{grid_size}m.laz"))

    # Resize original file
    laz.points = laz.points[mask_valid]
    laz.write(tile_src)
    if verbose:
        print("Saved file: ", tile_src)


def flattening(src_tiles, grid_size=10, verbose=True, verbose_full=False):
    print("Starting flattening:")
    list_tiles = [x for x in os.listdir(src_tiles) if x.endswith('.laz')]
    for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles), desc="Processing", disable=verbose==False):
        flattening_tile(
            tile_src=os.path.join(src_tiles, tile), 
            grid_size=grid_size,
            verbose=verbose_full,
            )
    

if __name__ == "__main__":
    tiles_src = r"D:\PDM_repo\Github\PDM\data\dataset_pipeline\tiles_20_flatten"
    flattening(tiles_src)
