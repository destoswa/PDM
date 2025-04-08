import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import pdal
import json
import laspy
import subprocess
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from omegaconf import OmegaConf
from src.format_conversions import convert_all_in_folder
from src.pseudo_labels_creation import update_attribute_where_cluster_match
# from models.KDE_classifier.inference import inference


class Pipeline():
    def __init__(self, cfg):
        self.cfg = cfg
        self.project_root_src = cfg.dataset.project_root_src
        self.data_src = cfg.dataset.data_src
        self.file_format = cfg.dataset.file_format
        self.segmenter_data_preds = cfg.segmenter.inference.data_preds
        self.cfg_classifier = cfg.classifier
        self.cfg_segmenter = cfg.segmenter
        self.segmenter_data_src = cfg.segmenter.data_src
        self.src_segment_model = cfg.segmenter.root_model_src
        self.loop_iteration = cfg.pipeline.loop_iteration
        self.src_preds_segmenter = None
        self.src_preds_classifier = None
        self.classified_clusters = None
        self.model_checkpoint_src = cfg.segmenter.inference.model_checkpoint_src

    @staticmethod
    def unzip_laz_files(zip_path, extract_to=".", delete_zip=True):
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
        if delete_zip:
            os.remove(zip_path)

    @staticmethod
    def run_subprocess(src_script, script_name, params=None, verbose=True):
        # go at the root of the segmenter
        old_current_dir = os.getcwd()
        os.chdir(src_script)

        # construct command and run subprocess
        script_str = script_name
        if params:
            script_str += ' ' + ' '.join([str(x) for x in params])
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

            if realtime_output and verbose:
                print(realtime_output.strip(), flush=True)

        # Ensure the process has fully exited
        process.wait()
        print(f"[INFO] Process completed with exit code {process.returncode}")

        # go back to original working dir
        os.chdir(old_current_dir)

    @staticmethod
    def split_instance(src_file_in, keep_ground=False, verbose=True):
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

    @staticmethod
    def change_var_val_yaml(src_yaml, var, val):
        # load yaml file
        yaml_raw = OmegaConf.load(src_yaml)
        yaml = OmegaConf.create(yaml_raw)  # now data.first_subsampling works
        
        # find the correct variable to change
        keys = var.split('/')
        d = yaml
        for key in keys[:-1]:
            d = d.setdefault(key, {})  # ensures intermediate keys exist

        # change value
        d[keys[-1]] = val  # set the new value
        
        # save back to yaml file
        OmegaConf.save(yaml, src_yaml)
        
    def segment(self):
        # self.src_preds_segmenter = os.path.normpath(self.segmenter_data_preds) + "/preds/"
        # return
        print("Starting inference:")

        # select checkpoint
        print(self.model_checkpoint_src)
        if self.model_checkpoint_src == "None":
            Pipeline.change_var_val_yaml(
                src_yaml=self.cfg_segmenter.inference.config_eval_src,
                var="checkpoint_dir",
                val="/home/pdm/models/SegmentAnyTree/model_file",
            )
        else:
            Pipeline.change_var_val_yaml(
                src_yaml=self.cfg_segmenter.inference.config_eval_src,
                var="checkpoint_dir",
                val=os.path.join(self.cfg.pipeline.root_src, self.model_checkpoint_src),
            )
        # return
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
        self.src_preds_segmenter = os.path.normpath(self.segmenter_data_preds) + "/preds/"
        print("Segmentation done!")
    
    def classify(self):
        print("Starting classification:")
        # inference(self.arg_classifier)
        # self.source_preds_classifier = self.arg.classifier.data_src
        file_format = self.cfg.dataset.file_format

        # loop on files:
        for file in os.listdir(self.segmenter_data_preds):
            if file.endswith(self.cfg.dataset.file_format):
                self.split_instance(os.path.join(self.segmenter_data_preds, file))

                # convert instances to pcd
                dir_target = os.path.dirname(os.path.join(self.segmenter_data_preds, file)) + '/' + os.path.join(self.cfg.segmenter.data_preds, file).split('/')[-1].split('.')[0] + "_split_instance"
                convert_all_in_folder(
                    src_folder_in=dir_target, 
                    src_folder_out=os.path.normpath(dir_target) + "/data", 
                    in_type=file_format, 
                    out_type='pcd'
                    )
                
                # makes predictions
                input_folder = os.path.normpath(self.project_root_src) + '/' + dir_target + '/'
                output_folder = os.path.normpath(self.project_root_src) + '/' + os.path.normpath(dir_target) + "/data/"
                self.run_subprocess(
                    src_script="/home/pdm/models/KDE_classifier/",
                    script_name="./run_inference.sh",
                    params= [input_folder, output_folder],
                    verbose=True
                    )
                
                # convert predictions to laz
                convert_all_in_folder(src_folder_in=output_folder, src_folder_out=output_folder, in_type='pcd', out_type='laz')
                for file in os.listdir(output_folder):
                    if file.endswith('.pcd'):
                        os.remove(os.path.join(output_folder, file))
                
    def create_pseudo_labels(self):        
        for child in os.listdir(self.segmenter_data_preds):
            # select only subfolders that coorespond to instances
            full_path = os.path.abspath(os.path.join(self.segmenter_data_preds, child))
            if os.path.isdir(full_path) and full_path.endswith('instance'):
                original_file_src = os.path.join(
                    self.data_src,
                    child.split('_out')[0] + '.' + self.file_format
                )
                results_src = os.path.join(full_path, 'results/')

                # load sources
                df_results = pd.read_csv(os.path.join(results_src, 'results.csv'), sep=';')
                # self.classified_clusters = pd.read_csv(os.path.join(results_src, 'results.csv'), sep=';')
                
                original_file = laspy.read(original_file_src)
                coords_A = np.stack((original_file.x, original_file.y, original_file.z), axis=1)
                coords_original_file_view = coords_A.view([('', coords_A.dtype)] * 3).reshape(-1)
            
                # add pseudo-label attribute if non-existant
                if not 'pseudo_label' in list(original_file.point_format.dimension_names):
                    original_file.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name='pseudo_label',
                            type=np.uint8,
                            description='pseudo-label'
                        )
                    )
                    original_file['pseudo_label'] = np.zeros(len(original_file), dtype=np.uint8)
                self.classified_clusters = []
                
                # load all clusters
                for _, row in df_results.iterrows():
                    cluster_path = os.path.join(full_path, row.file_name.split('.pcd')[0] + '.' + self.file_format)
                    cluster = laspy.open(cluster_path, mode='r').read()
                    coords_B = np.stack((cluster.x, cluster.y, cluster.z), axis=1)
                    coords_B_view = coords_B.view([('', coords_B.dtype)] * 3).reshape(-1)
                    self.classified_clusters.append((coords_B_view, row['class']))

                # create masks on the original tile for each cluster (multiprocessing)
                with ProcessPoolExecutor() as executor:
                    partialFunc = partial(self.process_row, coords_original_file_view)
                    results = list(tqdm(executor.map(partialFunc, range(len(self.classified_clusters))), total=len(self.classified_clusters), smoothing=0.9, desc="Updating pseudo-label"))
                
                # Update the original file based on results
                for mask, value in results:
                    original_file.pseudo_label[mask] = value
                
                # saving back original file
                original_file.write(original_file_src)

    def process_row(self, coords_original_file_view, row_id):
        # Find matching points between original file and cluster
        mask = np.isin(coords_original_file_view, self.classified_clusters[row_id][0])
        
        return mask, self.classified_clusters[row_id][1]

    def train(self):
        # modify dataset config file
        Pipeline.change_var_val_yaml(
            src_yaml=self.cfg_segmenter.training.config_data_src,
            var='data/dataroot',
            val=os.path.join(self.cfg.pipeline.root_src, self.cfg_segmenter.training.data_train),
        )

        # modify results directory
        if self.cfg_segmenter.training.result_src_full_name == 'None':
            self.cfg_segmenter.training.result_src_full_name = datetime.now().strftime(r"%Y%m%d_%H%M%S") + self.cfg_segmenter.training.result_src_name_suffixe
        Pipeline.change_var_val_yaml(
            src_yaml=self.cfg_segmenter.training.config_results_src,
            var='hydra/run/dir',
            # val="../../" + os.path.normpath(self.cfg_segmenter.training.result_training_dir) + "/" + r"${now:%Y%m%d_%H%M%S}_" + self.cfg_segmenter.training.result_src_name + '/' + str(self.loop_iteration),
            val="../../" + os.path.normpath(self.cfg_segmenter.training.result_training_dir) + "/" + self.cfg_segmenter.training.result_src_full_name + '/' + str(self.loop_iteration),
        )

        # run training script
        self.run_subprocess(
            src_script="/home/pdm/models/SegmentAnyTree/",
            script_name="./run_pipeline.sh",
            params= [102, 2],
            )
        
        # update path to checkpoint
        self.model_checkpoint_src = os.path.join(self.cfg.pipeline.root_src, os.path.normpath(self.cfg_segmenter.training.result_training_dir) + "/" + self.cfg_segmenter.training.result_src_full_name)
        