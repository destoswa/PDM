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


def remove_duplicates(laz_file, decimals=2):
    """
    Removes duplicate points from a LAS/LAZ file based on rounded 3D coordinates.

    Args:
        - laz_file (laspy.LasData): Input LAS/LAZ file as a laspy object.
        - decimals (int, optional): Number of decimals to round the coordinates for duplicate detection. Defaults to 2.

    Returns:
        - laspy.LasData: A new laspy object with duplicate points removed.
    """
        
    coords = np.round(np.vstack((laz_file.x, laz_file.y, laz_file.z)).T, decimals)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    mask = np.zeros(len(coords), dtype=bool)
    mask[unique_indices] = True

    # Create new LAS object
    header = laspy.LasHeader(point_format=laz_file.header.point_format, version=laz_file.header.version)
    new_las = laspy.LasData(header)

    for dim in laz_file.point_format.dimension_names:
        setattr(new_las, dim, getattr(laz_file, dim)[mask])

    return new_las


def flattening_tile(tile_src, tile_new_original_src, grid_size=10, verbose=True):
    """
    Flattens a tile by interpolating the ground surface and subtracting it from the original elevation.

    Args:
        - tile_src (str): Path to the input tile in LAS/LAZ format.
        - tile_new_original_src (str): Path to save the resized original tile after filtering.
        - grid_size (int, optional): Size of the grid in meters for local interpolation. Defaults to 10.
        - verbose (bool, optional): Whether to display progress and debug information. Defaults to True.

    Returns:
        - None: Saves the floor and flattened versions of the tile and updates the original file.
    """

    # Load file
    laz = laspy.read(tile_src)
    init_len = len(laz)
    laz = remove_duplicates(laz)
    if verbose:
        print(f"Removing duplicates: From {init_len} to {len(laz)}")
    laz.write(tile_new_original_src)
    
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
    laz.write(tile_new_original_src)
    if verbose:
        print("Saved file: ", tile_new_original_src)


def flattening(src_tiles, src_new_tiles, grid_size=10, verbose=True, verbose_full=False):
    """
    Applies the flattening process to all tiles in a directory using grid-based ground surface estimation.

    Args:
        - src_tiles (str): Path to the directory containing original tiles.
        - src_new_tiles (str): Path to the directory where resized tiles will be saved.
        - grid_size (int, optional): Size of the grid in meters for interpolation. Defaults to 10.
        - verbose (bool, optional): Whether to show a general progress bar. Defaults to True.
        - verbose_full (bool, optional): Whether to print detailed info per tile. Defaults to False.

    Returns:
        - None: Processes and saves flattened tiles into their respective folders.
    """
    
    print("Starting flattening:")
    list_tiles = [x for x in os.listdir(src_tiles) if x.endswith('.laz')]
    for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles), desc="Processing", disable=verbose==False):
        if verbose_full:
            print("Flattening tile: ", tile)
        flattening_tile(
            tile_src=os.path.join(src_tiles, tile), 
            tile_new_original_src=os.path.join(src_new_tiles, tile),
            grid_size=grid_size,
            verbose=verbose_full,
            )
    

if __name__ == "__main__":
    tiles_src = r"D:\PDM_repo\Github\PDM\data\dataset_pipeline\tiles_20_flatten"
    flattening(tiles_src)
