import os
import numpy as np
import laspy
from tqdm import tqdm
import pdal
import json


def split_instance(src_file_in, path_out="", keep_ground=False, verbose=True):
    # Define target folder:
    if path_out == "":
        dir_target = os.path.join(os.path.dirname(src_file_in), os.path.basename(src_file_in).split('.')[0] + "_split_instance")
    else:
        dir_target = os.path.join(path_out, os.path.basename(src_file_in).split('.')[0] + "_split_instance")

    if not os.path.exists(dir_target):
        os.makedirs(dir_target)

    points_segmented = laspy.read(src_file_in)

    for idx, instance in tqdm(enumerate(set(points_segmented.PredInstance)), total=len(set(points_segmented.PredInstance)), disable=~verbose):
        if not keep_ground and instance == 0:
            continue
        file_name = src_file_in.split('\\')[-1].split('/')[-1].split('.laz')[0] + f'_{instance}.laz'
        src_instance = os.path.join(dir_target, file_name)

        las = laspy.read(src_file_in)

        pred_instance = las['PredInstance']

        mask = pred_instance == instance

        filtered_points = las.points[mask]

        # new_header = las.header.copy()

        filtered_las = laspy.LasData(las.header)
        filtered_las.points = filtered_points

        # Preserve CRS if it exists
        # crs = las.header.parse_crs()
        # if crs:
        #     filtered_las.header.parse_crs().wkt = crs.to_wkt()

        # Write the filtered file
        filtered_las.write(src_instance)

        # # Define the PDAL pipeline for filtering
        # # Step 1: Fix LAS header and write to a temporary file
        # fix_pipeline_json = {
        #     "pipeline": [
        #         {
        #             "type": "readers.las",
        #             "filename": src_file_in,
        #             "override_srs": "EPSG:2056"  # or whatever CRS your data uses
        #         },
        #         {
        #             "type": "writers.las",
        #             "filename": "temp_fixed.laz",
        #             "global_encoding": 1
        #         }
        #     ]
        # }
        # fix_pipeline = pdal.Pipeline(json.dumps(fix_pipeline_json))
        # fix_pipeline.execute()

        # # Step 2: Apply expression filter on the fixed file
        # filter_pipeline_json = {
        #     "pipeline": [
        #         "temp_fixed.laz",
        #         {
        #             "type": "filters.expression",
        #             "expression": f"PredInstance == {instance}"
        #         },
        #         file_src
        #     ]
        # }

        # filter_pipeline = pdal.Pipeline(json.dumps(filter_pipeline_json))
        # filter_pipeline.execute()
        # # Run PDAL pipeline
        # # pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        # # pipeline.execute()

    if verbose:
        print(f"INSTANCE SPLITTING DONE on {src_file_in}")


def split_semantic(src, verbose=True):
    # Define target folder:
    dir_target = os.path.dirname(src) + '/' + src.split('/')[-1].split('.')[0] + "_split_semantic"

    if not os.path.exists(dir_target):
        os.makedirs(dir_target)

    points_segmented = laspy.read(src)
    val_to_name = ['ground', 'tree']

    for val, name in enumerate(val_to_name):
        file_name = src.split('\\')[-1].split('/')[-1].split('.laz')[0] + f'_{name}.laz'
        file_src = os.path.join(dir_target, file_name)

        # Define the PDAL pipeline for filtering
        pipeline_json = {
            "pipeline": [
                src,
                {
                    "type": "filters.expression",
                    "expression": f"PredSemantic == {val}"
                },
                file_src
            ]
        }

        # Run PDAL pipeline
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        
    if verbose:
        print("SEMANTIC SPLITTING DONE")
