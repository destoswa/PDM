import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import pdal
import json
import laspy
import subprocess
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from omegaconf import OmegaConf
from src.format_conversions import convert_all_in_folder
from src.pseudo_labels_creation import update_attribute_where_cluster_match
from scipy.spatial import cKDTree
from src.metrics import compute_classification_results, compute_panoptic_quality, compute_mean_iou
# from models.KDE_classifier.inference import inference


class Pipeline():
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_src = cfg.dataset.project_root_src
        self.classifier_root_src = cfg.classifier.root_model_src
        self.segmenter_root_src = cfg.segmenter.root_model_src

        # config regarding dataset
        self.data_src = cfg.dataset.data_src
        self.file_format = cfg.dataset.file_format
        self.tiles_all = [file for file in os.listdir(self.data_src) if file.endswith(self.file_format)]
        self.tiles_to_process = self.tiles_all.copy()

        # config regarding pipeline
        self.num_loops = cfg.pipeline.loop_iteration
        self.current_loop = 0
        self.results_src = ""
        self.upgrade_ground = cfg.pipeline.processes.upgrade_ground
        self.garbage_as_grey = cfg.pipeline.processes.garbage_as_grey

        # config regarding inference
        self.inference = cfg.segmenter.inference
        self.preds_src = os.path.join(self.data_src, 'preds')
        self.problematic_tiles = []
        self.empty_tiles = []

        # config regarding classification
        self.classification = cfg.classifier
        self.src_preds_classifier = None
        self.classified_clusters = None

        # config regarding training
        self.training = cfg.segmenter.training
        self.model_checkpoint_src = cfg.segmenter.inference.model_checkpoint_src
        self.current_epoch = 0
        self.training.result_src_name = datetime.now().strftime(r"%Y%m%d_%H%M%S_") + self.training.result_src_name_suffixe
        # self.training.result_src_name = "20250417_133119_test"
        self.result_dir = os.path.join(self.root_src, self.training.result_training_dir, self.training.result_src_name)
        self.result_pseudo_labels_dir = os.path.join(self.result_dir, 'pseudo_labels/')

        #   _create result dirs
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.result_pseudo_labels_dir, exist_ok=True)

        #   _copy files
        print("Copying files")
        for _, file in tqdm(enumerate(self.tiles_to_process), total=len(self.tiles_to_process), desc="Process"):
            shutil.copyfile(
                os.path.join(self.data_src, file),
                os.path.join(self.result_pseudo_labels_dir, file)
            )

        # config regarding metrics
        #   _ training metrics
        training_metrics_columns = [
            'num_loop', 'num_epoch', 'stage', 'loss', 'offset_norm_loss', 
            'offset_dir_loss', 'ins_loss', 'ins_var_loss', 'ins_dist_loss', 
            'ins_reg_loss', 'semantic_loss', 'score_loss', 'acc', 'macc', 
            'mIoU', 'pos', 'neg', 'Iacc', 'cov', 'wcov', 'mIPre', 'mIRec', 
            'F1', 'map',
            ]
        self.training_metrics = pd.DataFrame(columns=training_metrics_columns)
        self.training_metrics_src = os.path.join(self.root_src, self.training.result_training_dir, self.training.result_src_name, 'training_metrics.csv')
        self.training_metrics.to_csv(self.training_metrics_src, sep=';', index=False)

        #   _ inference metrics
        inference_metrics_columns = [
            'name', 'num_loop', 'is_problematic', 'is_empty', 'num_predictions', 
            'num_garbage', 'num_multi', 'num_single', 'PQ', 'SQ', 'RQ', 'Pre', 
            'Rec', 'mIoU',
            ]
        self.inference_metrics = pd.DataFrame(columns=inference_metrics_columns)
        self.inference_metrics_src = os.path.join(self.root_src, self.training.result_training_dir, self.training.result_src_name, 'inference_metrics.csv')
        self.inference_metrics.to_csv(self.inference_metrics_src, sep=';', index=False)

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
        if verbose:
            print(f"[INFO] Process completed with exit code {process.returncode}")

        # go back to original working dir
        os.chdir(old_current_dir)

        # return exit code
        return process.returncode

    @staticmethod
    def split_instance(src_file_in, keep_ground=False, verbose=True):
        # Define target folder:
        dir_target = os.path.dirname(src_file_in) + '/' + src_file_in.split('/')[-1].split('.')[0] + "_split_instance"

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

    @staticmethod
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

        # laz_file.write(laz_file_src)

    @staticmethod
    def match_pointclouds(laz1, laz2):
        """Sort laz2 to match the order of laz1 without changing laz1's order.

        Args:
            laz1: laspy.LasData object (reference order)
            laz2: laspy.LasData object (to be sorted)
        
        Returns:
            laz2 sorted to match laz1
        """
        # Retrieve and round coordinates for robust matching
        coords_1 = np.round(np.vstack((laz1.x, laz1.y, laz1.z)), 2).T
        coords_2 = np.round(np.vstack((laz2.x, laz2.y, laz2.z)), 2).T

        # Verify laz2 is of the same size as laz1
        assert len(coords_2) == len(coords_1), "laz2 should be a subset of laz1"

        # Create a dictionary mapping from coordinates to indices
        coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coords_1)}

        # Find indices in laz1 that correspond to laz2
        matching_indices = []
        failed = 0
        for coord in coords_2:
            try:
                matching_indices.append(coord_to_idx[tuple(coord)])
            except Exception as e:
                failed += 1

        matching_indices = np.array([coord_to_idx[tuple(coord)] for coord in coords_2])

        # Sort laz2 to match laz1
        sorted_indices = np.argsort(matching_indices)

        # Apply sorting to all attributes of laz2
        laz2.points = laz2.points[sorted_indices]

        return laz2  # Now sorted to match laz1

    def segment(self, verbose=False):
        print("Starting inference:")
        if not os.path.exists(self.preds_src):
            os.mkdir(self.preds_src)

        # select checkpoint
        if self.model_checkpoint_src == "None":
            Pipeline.change_var_val_yaml(
                src_yaml=self.inference.config_eval_src,
                var="checkpoint_dir",
                val="/home/pdm/models/SegmentAnyTree/model_file",
            )
        else:
            Pipeline.change_var_val_yaml(
                src_yaml=self.inference.config_eval_src,
                var="checkpoint_dir",
                val=os.path.join(self.cfg.pipeline.root_src, self.model_checkpoint_src),
            )

        # create temp folder
        temp_seg_src = os.path.join(os.path.join(self.root_src, self.data_src), 'temp_seg/')
        if os.path.exists(temp_seg_src):
            shutil.rmtree(temp_seg_src)
        os.mkdir(temp_seg_src)



        # list_files = [x for x in os.listdir(os.path.join(self.root_src, self.data_src)) if x.endswith(self.file_format) and x not in self.problematic_tiles]
        # if self.classification.processes.do_remove_empty_tiles:
        #     list_files = [x for x in list_files if x not in self.empty_tiles]

        # for _, file in tqdm(enumerate(list_files), total=len(list_files), desc="Processing"):



        # creates pack of samples to infer on
        list_pack_of_tiles = [self.tiles_to_process[x:min(y,len(self.tiles_to_process))] for x, y in zip(
            range(0, len(self.tiles_to_process) - self.inference.num_tiles_per_inference, self.inference.num_tiles_per_inference),
            range(self.inference.num_tiles_per_inference, len(self.tiles_to_process), self.inference.num_tiles_per_inference),
            )]
        if list_pack_of_tiles[-1][-1] != self.tiles_to_process[-1]:
            list_pack_of_tiles.append(self.tiles_to_process[(len(list_pack_of_tiles)*self.inference.num_tiles_per_inference)::])
        
        # loops on samples:
        for _, file in tqdm(enumerate(self.tiles_to_process), total=len(self.tiles_to_process), desc="Processing"):
            if verbose:
                print("===\tProcessing file: ", file, "\t===")
            # original_file_src = os.path.join(os.path.join(self.root_src, self.data_src, file))
            original_file_src = os.path.join(os.path.join(self.result_pseudo_labels_dir, self.data_src, file))
            temp_file_src = os.path.join(os.path.join(temp_seg_src, file))
            shutil.copyfile(
                original_file_src,
                temp_file_src,
            )
            
            return_code = self.run_subprocess(
                src_script=self.segmenter_root_src,
                script_name="./run_oracle_pipeline.sh",
                params= [temp_seg_src, temp_seg_src],
                verbose=verbose
                )
        
            # catch errors
            if return_code != 0:
                if verbose:
                    print(f"Problem with tile {file}")
                self.problematic_tiles.append(file)
            else:
                # unzip results
                if verbose:
                    print("Unzipping results...")
                self.unzip_laz_files(
                    zip_path=os.path.join(temp_seg_src, "results.zip"),
                    extract_to=self.preds_src,
                    delete_zip=True
                    )

                # check if empty
                src_pred_file = os.path.join(self.preds_src, file.split('.')[0] + '_out.' + self.file_format)
                tile = laspy.read(src_pred_file)
                num_instances = len(set(tile.PredInstance))
                if num_instances == 1:
                    if verbose:
                        print(f"Empty tile: {file}")
                    self.empty_tiles.append(file)
                    if self.classification.processes.do_remove_empty_tiles:
                        os.remove(os.path.join(src_pred_file))
                        # if file in self.tiles_to_process:
                        #     self.tiles_to_process.remove(file)
                if verbose:
                    print("Segmentation done!")

            # removing temp file
            os.remove(temp_file_src)

        # update tiles to process
        for tile in self.problematic_tiles:
            if tile in self.tiles_to_process:
                self.tiles_to_process.remove(tile)
        if self.classification.processes.do_remove_empty_tiles:
            for tile in self.empty_tiles:
                if tile in self.tiles_to_process:
                    self.tiles_to_process.remove(tile)

        # removing temp folder
        shutil.rmtree(temp_seg_src)

        # update segmentation state
        if len(self.problematic_tiles) > 0:
            print("========\nProblematic tiles:")
            for f in self.problematic_tiles:
                print("\t- ", f)
                self.inference_metrics.loc[(self.inference_metrics.name == f) & (self.inference_metrics.num_loop == self.current_loop), "is_problematic"] = 1
        if len(self.empty_tiles) > 0:
            print("========\nEmpty tiles:")
            for f in self.empty_tiles:
                print("\t- ", f)
                self.inference_metrics.loc[(self.inference_metrics.name == f) & (self.inference_metrics.num_loop == self.current_loop), "is_empty"] = 1
    
    def classify(self, verbose=False):
        print("Starting classification:")

        # loop on files:
        list_files = [f for f in os.listdir(self.preds_src) if f.endswith(self.file_format)]
        for _, file in tqdm(enumerate(list_files), total=len(list_files), desc="Processing"):
            self.split_instance(os.path.join(self.preds_src, file), verbose=verbose)

            # convert instances to pcd
            dir_target = self.preds_src + '/' + file.split('/')[-1].split('.')[0] + "_split_instance"
            convert_all_in_folder(
                src_folder_in=dir_target, 
                src_folder_out=os.path.normpath(dir_target) + "/data", 
                in_type=self.file_format, 
                out_type='pcd',
                verbose=verbose
                )
            
            # makes predictions
            input_folder = dir_target
            output_folder = os.path.join(dir_target, 'data')
            code_return = self.run_subprocess(
                src_script=self.classifier_root_src,
                script_name="./run_inference.sh",
                params= [input_folder, output_folder],
                verbose=verbose
                )
            if code_return != 0:
                print(f"WARNING! Subprocess for classification return code {code_return}!!")
            
            # convert predictions to laz
            convert_all_in_folder(src_folder_in=output_folder, src_folder_out=output_folder, in_type='pcd', out_type='laz')

            # remove pcd files
            for file in os.listdir(output_folder):
                if file.endswith('.pcd'):
                    os.remove(os.path.join(output_folder, file))
            
    def create_pseudo_labels(self, verbose=False):
        print("Creating pseudo labels:")
        list_folders = [x for x in os.listdir(self.preds_src) if os.path.abspath(os.path.join(self.preds_src, x)) and x.endswith('instance')]
        for _, child in tqdm(enumerate(list_folders), total=len(list_folders), desc='Processing'):
            # load original file
            # original_file_src = os.path.join(self.data_src, child.split('_out')[0] + '.' + self.file_format)
            original_file_src = os.path.join(self.result_pseudo_labels_dir, child.split('_out')[0] + '.' + self.file_format)
            original_file = laspy.read(original_file_src)

            # load prediction file
            pred_file_src = os.path.join(self.data_src, 'preds', child.split('_split')[0] + '.' + self.file_format)
            pred_file = laspy.read(pred_file_src)
            
            # create pseudo-labeled file
            new_file = laspy.read(original_file_src)
            
            # load sources
            full_path = os.path.abspath(os.path.join(self.preds_src, child))
            results_src = os.path.join(full_path, 'results/')
            df_results = pd.read_csv(os.path.join(results_src, 'results.csv'), sep=';')
            
            # match the original with the pred
            Pipeline.remove_duplicates(new_file)
            Pipeline.remove_duplicates(pred_file)
            Pipeline.match_pointclouds(new_file, pred_file)

            coords_A = np.stack((original_file.x, original_file.y, original_file.z), axis=1)
            coords_original_file_view = coords_A.view([('', coords_A.dtype)] * 3).reshape(-1)

            # add pseudo-labels attribute if non-existant
            #   _add instance
            if 'treeID' not in list(original_file.point_format.dimension_names):
                new_file.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name='treeID',
                        type="f4",
                        description='Instance p-label'
                    )
                )
            
                # reset values of pseudo-labels
                new_file.treeID = np.zeros(len(new_file), dtype="f4")
                new_file.classification = np.zeros(len(new_file), dtype="f4")

            # set ground based on semantic pred
            if self.upgrade_ground or self.current_loop == 0:
                new_file.classification[pred_file.PredSemantic == 0] = 1
                
            # load all clusters
            self.classified_clusters = []
            for _, row in df_results.iterrows():
                cluster_path = os.path.join(full_path, row.file_name.split('.pcd')[0] + '.' + self.file_format)
                cluster = laspy.open(cluster_path, mode='r').read()
                coords_B = np.stack((cluster.x, cluster.y, cluster.z), axis=1)
                coords_B_view = coords_B.view([('', coords_B.dtype)] * 3).reshape(-1)
                if row['class'] in [1, 2]:
                    self.classified_clusters.append((coords_B_view, row['class']))

            # create masks on the original tile for each cluster (multiprocessing)
            with ProcessPoolExecutor() as executor:
                partialFunc = partial(self.process_row, coords_original_file_view)
                results = list(tqdm(executor.map(partialFunc, range(len(self.classified_clusters))), total=len(self.classified_clusters), smoothing=0.9, desc="Updating pseudo-label", disable=~verbose))
            
            # Update the original file based on results and update the csv file with ref to trees
            for id_tree, (mask, value) in tqdm(enumerate(results), total=len(results), desc='temp', disable=~verbose):
                """
                value to label:
                    0: garbage
                    1: multi
                    2: single
                classification to label:
                    0: grey
                    1: ground
                    4: tree
                """
                # set instance p-label
                new_file.treeID[mask] = id_tree
                
                # set semantic p-label
                if value == 1 or (self.garbage_as_grey and value == 0):
                    new_file.classification[mask] = 0
                elif value == 0 and not self.garbage_as_grey:
                    new_file.classification[mask] = 1
                elif value == 2:
                    new_file.classification[mask] = 4
                else:
                    raise ValueError("Problem in the setting of the semantic pseudo-labels!")
                
            # saving back original file
            new_file.write(original_file_src)

    def process_row(self, coords_original_file_view, row_id):
        # Find matching points between original file and cluster
        mask = np.isin(coords_original_file_view, self.classified_clusters[row_id][0])
        
        return mask, self.classified_clusters[row_id][1]

    def stats_on_tiles(self):
        print("Computing stats on tiles")
        # lst_files = [x for x in os.listdir(self.data_src) if x.endswith(self.file_format) and x not in self.problematic_tiles]
        # if self.classification.processes.do_remove_empty_tiles and len(self.empty_tiles) > 0:
        #     lst_files = [f for f in lst_files if f not in self.empty_tiles]

        # for _, file in tqdm(enumerate(lst_files), total=len(lst_files), desc="processing"):
        for _, file in tqdm(enumerate(self.tiles_to_process), total=len(self.tiles_to_process), desc="processing"):
            # Add stats to state variable
            dir_target = self.preds_src + '/' + file.split('/')[-1].split('.')[0] + "_out_split_instance"
            [num_garbage, num_multi, num_single] = compute_classification_results(os.path.join(dir_target, 'results'))
            self.inference_metrics.loc[(self.inference_metrics.name == file) & (self.inference_metrics.num_loop == self.current_loop), "num_garbage"] = num_garbage
            self.inference_metrics.loc[(self.inference_metrics.name == file) & (self.inference_metrics.num_loop == self.current_loop), "num_multi"] = num_multi
            self.inference_metrics.loc[(self.inference_metrics.name == file) & (self.inference_metrics.num_loop == self.current_loop), "num_single"] = num_single
            self.inference_metrics.loc[(self.inference_metrics.name == file) & (self.inference_metrics.num_loop == self.current_loop), "num_predictions"] = num_garbage + num_multi + num_single

            # metrics on pseudo labels
            tile_original = laspy.read(os.path.join(self.result_pseudo_labels_dir, file))
            tile_preds = laspy.read(os.path.join(self.data_src, 'preds', file.split('.')[0] + '_out.' + self.file_format))

            # match the original with the pred
            Pipeline.remove_duplicates(tile_original)
            Pipeline.remove_duplicates(tile_preds)
            Pipeline.match_pointclouds(tile_original, tile_preds)

            gt_instances = tile_original.treeID
            pred_instances = tile_preds.PredInstance
            PQ, SQ, RQ, tp, fp, fn = compute_panoptic_quality(gt_instances, pred_instances)
            metrics = {
                'PQ': PQ,
                'SQ': SQ,
                'RQ': RQ,
                'Rec': round(tp/(tp + fn), 2) if tp + fn > 0 else 0,
                'Pre': round(tp/(tp + fp),2) if tp + fp > 0 else 0,
            }
            for metric_name, metric_val in metrics.items():
                self.inference_metrics.loc[(self.inference_metrics.name == file) & (self.inference_metrics.num_loop == self.current_loop), metric_name] = metric_val

    def prepare_data(self):
        print("Prepare data:")
        self.run_subprocess(
            src_script="/home/pdm/models/SegmentAnyTree/",
            script_name="./run_sample_data_conversion.sh",
            # params= [self.data_src],
            params= [self.result_pseudo_labels_dir],
            )

    def train(self):
        print("Training:")
        # modify dataset config file
        Pipeline.change_var_val_yaml(
            src_yaml=self.training.config_data_src,
            var='data/dataroot',
            val=os.path.join(self.result_pseudo_labels_dir),
        )

        # modify results directory
        Pipeline.change_var_val_yaml(
            src_yaml=self.training.config_results_src,
            var='hydra/run/dir',
            val="../../" + os.path.normpath(self.training.result_training_dir) + "/" + self.training.result_src_name + '/' + str(self.current_loop),
        )

        # run training script
        self.run_subprocess(
            src_script="/home/pdm/models/SegmentAnyTree/",
            script_name="./run_pipeline.sh",
            params= [self.training.num_trainings_per_loop, 
                     self.training.batch_size, 
                     self.training.sample_per_epoch,
                     self.training_metrics_src,
                     self.current_loop,],
            )
        
        # update path to checkpoint
        self.model_checkpoint_src = os.path.join(self.cfg.pipeline.root_src, os.path.normpath(self.training.result_training_dir) + "/" + self.training.result_src_name + '/' + str(self.current_loop))
        