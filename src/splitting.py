import os
import numpy as np
import laspy
from tqdm import tqdm
# import pdal
# import json


def split_instance(src_file_in, path_out="", keep_ground=False, verbose=True):
    """
    Split a LAS/LAZ file into separate files based on PredInstance values.

    Parameters:
    - src_file_in (str): Path to the input LAS/LAZ file.
    - path_out (str): Optional target directory. Default creates a sibling folder.
    - keep_ground (bool): If False, ignores instance 0. Default is False.
    - verbose (bool): If True, print progress. Default is True.

    Returns:
    - None
    """
    
    # Define target folder:
    if path_out == "":
        dir_target = os.path.join(os.path.dirname(src_file_in), os.path.basename(src_file_in).split('.')[0] + "_split_instance")
    else:
        dir_target = os.path.join(path_out, os.path.basename(src_file_in).split('.')[0] + "_split_instance")

    if not os.path.exists(dir_target):
        os.makedirs(dir_target)

    points_segmented = laspy.read(src_file_in)

    for _, instance in tqdm(enumerate(set(points_segmented.PredInstance)), total=len(set(points_segmented.PredInstance)), disable=~verbose):
        if not keep_ground and instance == 0:
            continue
        file_name = src_file_in.split('\\')[-1].split('/')[-1].split('.laz')[0] + f'_{instance}.laz'
        src_instance = os.path.join(dir_target, file_name)

        las = laspy.read(src_file_in)

        pred_instance = las['PredInstance']

        mask = pred_instance == instance

        filtered_points = las.points[mask]

        filtered_las = laspy.LasData(las.header)
        filtered_las.points = filtered_points

        # Write the filtered file
        filtered_las.write(src_instance)

    if verbose:
        print(f"INSTANCE SPLITTING DONE on {src_file_in}")


# def split_semantic(src, verbose=True):
#     # Define target folder:
#     dir_target = os.path.dirname(src) + '/' + src.split('/')[-1].split('.')[0] + "_split_semantic"

#     if not os.path.exists(dir_target):
#         os.makedirs(dir_target)

#     points_segmented = laspy.read(src)
#     val_to_name = ['ground', 'tree']

#     for val, name in enumerate(val_to_name):
#         file_name = src.split('\\')[-1].split('/')[-1].split('.laz')[0] + f'_{name}.laz'
#         file_src = os.path.join(dir_target, file_name)

#         # Define the PDAL pipeline for filtering
#         pipeline_json = {
#             "pipeline": [
#                 src,
#                 {
#                     "type": "filters.expression",
#                     "expression": f"PredSemantic == {val}"
#                 },
#                 file_src
#             ]
#         }

#         # Run PDAL pipeline
#         pipeline = pdal.Pipeline(json.dumps(pipeline_json))
#         pipeline.execute()
        
#     if verbose:
#         print("SEMANTIC SPLITTING DONE")
