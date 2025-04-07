import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import pdal
import json
import laspy
import subprocess
from tqdm import tqdm
from src.format_conversions import convert_all_in_folder
# from models.KDE_classifier.inference import inference


class Pipeline():
    def __init__(self, cfg):
        print(cfg.dataset.project_root_src)
        self.cfg = cfg
        self.project_root_src = cfg.dataset.project_root_src
        self.segmenter_data_src = cfg.segmenter.data_src
        self.segmenter_data_preds = cfg.segmenter.data_preds
        self.cfg_classifier = cfg.classifier
        self.cfg_segmenter = cfg.segmenter
        self.src_segment_model = cfg.segmenter.root_src
        self.src_preds_segmenter = None
        self.src_preds_classifier = None

    @staticmethod
    def unzip_laz_files(zip_path, extract_to="."):
        """Extract all .laz files from a zip archive to a target directory root.

        zip_path (str): Path to the .zip archive.
        extract_to (str): Directory where .laz files will be extracted. Defaults to current directory.
        """
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            laz_files = [f for f in zip_ref.namelist() if f.lower().endswith('.laz') and not f.endswith('/')]
            for file in laz_files:
                # Extract and flatten the path to root
                filename = os.path.basename(file)
                with zip_ref.open(file) as source, open(os.path.join(extract_to, filename), 'wb') as target:
                    target.write(source.read())

    @staticmethod
    def run_subprocess(src_script, script_name, params=None):
        # go at the root of the segmenter
        old_current_dir = os.getcwd()
        os.chdir(src_script)
        print(os.getcwd())

        # construct command and run subprocess
        script_str = script_name
        if params:
            script_str += ' ' + ' '.join(params)
        print(script_str)
        process = subprocess.Popen(
            script_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            encoding='utf-8',
            errors='replace'
        )

        while True:
            realtime_output = process.stdout.readline()

            if realtime_output == '' and process.poll() is not None:
                break

            if realtime_output:
                print(realtime_output.strip(), flush=True)

        # Ensure the process has fully exited
        process.wait()
        print(f"[INFO] Process completed with exit code {process.returncode}")

        # go back to original working dir
        os.chdir(old_current_dir)

    @staticmethod
    def split_instance(src_file_in, keep_ground=False, verbose=True):
        print("src_file", src_file_in)
        # Define target folder:
        dir_target = os.path.dirname(src_file_in) + '/' + src_file_in.split('/')[-1].split('.')[0] + "_split_instance"

        if not os.path.exists(dir_target):
            os.makedirs(dir_target)

        points_segmented = laspy.read(src_file_in)

        for idx, instance in tqdm(enumerate(set(points_segmented.PredInstance)), total=len(set(points_segmented.PredInstance))):
            if not keep_ground and instance == 0:
                continue
            file_name = src_file_in.split('\\')[-1].split('/')[-1].split('.laz')[0] + f'_{instance}.laz'
            src_instance = os.path.join(dir_target, file_name)
            print("src instance: ", src_instance)

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

    @staticmethod
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

    def segment(self):
        print("Starting inference:")
        self.run_subprocess(
            src_script="/home/pdm/models/SegmentAnyTree/",
            script_name="./run_oracle_pipeline.sh",
            params= [os.path.join(self.project_root_src, self.segmenter_data_src),
                     os.path.join(self.project_root_src, self.segmenter_data_preds),
                     ],
            )
        # unzip results
        print("Unzipping results...")
        self.unzip_laz_files(
            zip_path=os.path.join(self.project_root_src, self.segmenter_data_preds, "results.zip"),
            extract_to=os.path.normpath(self.project_root_src) + '/' + os.path.normpath(self.segmenter_data_preds),
            )
        # self.src_preds_segmenter = self.data_src
        print("Segmentation done!")
    
    def classify(self):
        # inference(self.arg_classifier)
        # self.source_preds_classifier = self.arg.classifier.data_src
        file_format = self.cfg.dataset.file_format

        # # prepare data
        # if file_format != 'pcd':
        #     print("Converting data into pcd..")
        #     convert_all_in_folder(
        #         src_folder_in=self.cfg.segmenter.data_preds, 
        #         src_folder_out=os.path.normpath(self.cfg.segmenter.data_preds) + "/converted", 
        #         in_type=file_format, 
        #         out_type='pcd'
        #         )
        #     data_src = os.path.normpath(self.cfg.segmenter.data_preds) + "/converted"
        #     self.cfg.dataset.data_src = data_src

        # loop on files:
        for file in os.listdir(self.cfg.segmenter.data_preds):
            if file.endswith(self.cfg.dataset.file_format):
                print(file)
                self.split_instance(os.path.join(self.cfg.segmenter.data_preds, file))

                # convert instances to pcd
                dir_target = os.path.dirname(os.path.join(self.cfg.segmenter.data_preds, file)) + '/' + os.path.join(self.cfg.segmenter.data_preds, file).split('/')[-1].split('.')[0] + "_split_instance"
                convert_all_in_folder(
                    src_folder_in=dir_target, 
                    src_folder_out=os.path.normpath(dir_target) + "/converted", 
                    in_type=file_format, 
                    out_type='pcd'
                    )
                
                # makes predictions
                self.run_subprocess(
                    src_script="/home/pdm/models/KDE_classifier/",
                    script_name="./run_inference.sh",
                    params= [os.path.normpath(self.project_root_src) + '/' + dir_target + '/',
                            os.path.normpath(self.project_root_src) + '/' + os.path.normpath(dir_target) + "/converted/",
                            ],
                    )


    def prepare_gt(self):
        assert self.source_preds_classifier != None
        
