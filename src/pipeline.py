import os
import sys
import shutil
import zipfile
import numpy as np
import pandas as pd
# import pdal
import json
import laspy
import subprocess
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
if __name__ == "__main__":
    sys.path.append(os.getcwd())
from src.format_conversions import convert_all_in_folder
from src.pseudo_labels_creation import update_attribute_where_cluster_match
from src.metrics import compute_classification_results, compute_panoptic_quality, compute_mean_iou
from src.visualization import show_global_metrics, show_inference_counts, show_inference_metrics
from src.splitting import split_instance
# from models.KDE_classifier.inference import inference


class Pipeline():
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_src = cfg.pipeline.root_src
        self.classifier_root_src = cfg.classifier.root_model_src
        self.segmenter_root_src = cfg.segmenter.root_model_src

        # config regarding dataset
        self.data_src = cfg.dataset.data_src
        self.file_format = cfg.dataset.file_format
        self.tiles_all = [file for file in os.listdir(self.data_src) if file.endswith(self.file_format)]
        self.tiles_to_process = self.tiles_all.copy()

        # config regarding pipeline
        self.num_loops = cfg.pipeline.num_loops
        self.current_loop = 0
        self.results_root_src = cfg.pipeline.results_root_src
        self.upgrade_ground = cfg.pipeline.processes.upgrade_ground
        self.garbage_as_grey = cfg.pipeline.processes.garbage_as_grey
        self.save_pseudo_labels_per_loop = cfg.pipeline.processes.save_pseudo_labels_per_loop
        self.do_continue_from_existing = cfg.pipeline.preload.do_continue_from_existing
        self.log = ""

        # config regarding inference
        self.inference = cfg.segmenter.inference
        self.preds_src = ""
        self.problematic_tiles = []
        self.empty_tiles = []

        # config regarding classification
        self.classification = cfg.classifier
        self.src_preds_classifier = None
        self.classified_clusters = None

        # config regarding training
        self.training = cfg.segmenter.training
        self.model_checkpoint_src = None
        self.current_epoch = 0

        # config regarding results
        self.result_src_name_suffixe = cfg.pipeline.result_src_name_suffixe
        self.result_src_name = datetime.now().strftime(r"%Y%m%d_%H%M%S_") + self.result_src_name_suffixe
        # self.training.result_src_name = "20250417_133119_test"
        # self.result_pseudo_labels_dir = f'results/trainings/{self.result_src_name}/pseudo_labels/'

        if self.do_continue_from_existing:
            self.result_dir = cfg.pipeline.preload.src_existing
            self.result_src_name = os.path.basename(self.result_dir)
            # find loop
            num_loop = 0
            while str(num_loop) in os.listdir(self.result_dir):
                num_loop += 1
            if num_loop == 0:
                raise ValueError("There is no existing loops in the project you are trying to start from!!")
            else:
                self.current_loop = num_loop
                
        #   _create result dirs if necessary
        self.result_dir = os.path.join(self.root_src, self.results_root_src, self.result_src_name)
        self.result_current_loop_dir = os.path.join(self.result_dir, str(self.current_loop))
        self.result_pseudo_labels_dir = os.path.join(self.result_dir, 'pseudo_labels/')

        # update model to use if starting from existing pipeline
        if self.do_continue_from_existing:
            pass
            # self.model_checkpoint_src = os.path.join(self.result_dir, str(self.current_loop - 1))

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.result_pseudo_labels_dir, exist_ok=True)
        os.makedirs(self.result_current_loop_dir, exist_ok=True)

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
        self.training_metrics_src = os.path.join(self.result_dir, 'training_metrics.csv')

        #   _ inference metrics
        inference_metrics_columns = [
            'name', 'num_loop', 'is_problematic', 'is_empty', 'num_predictions', 
            'num_garbage', 'num_multi', 'num_single', 'PQ', 'SQ', 'RQ', 'Pre', 
            'Rec', 'mIoU',
            ]
        self.inference_metrics = pd.DataFrame(columns=inference_metrics_columns)
        self.inference_metrics_src = os.path.join(self.result_dir, 'inference_metrics.csv')

        if not self.do_continue_from_existing:
            #   _copy files
            print("Copying files")
            for _, file in tqdm(enumerate(self.tiles_to_process), total=len(self.tiles_to_process), desc="Process"):
                shutil.copyfile(
                    os.path.join(self.data_src, file),
                    os.path.join(self.result_pseudo_labels_dir, file)
                )
                
            self.training_metrics.to_csv(self.training_metrics_src, sep=';', index=False)
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

    @staticmethod
    def transform_with_pca(pointcloud, verbose=False):
        # fit PCA
        pca = PCA(n_components=2)

        # compute pointcloud in new axes
        transformed = pca.fit_transform(pointcloud)

        # principal axes
        components = pca.components_  
        if verbose:
            print("PCA components (axes):\n", components)
            print("PCA-transformed points:\n", transformed)
        
        return transformed
    
    @staticmethod
    def split_instances(pointcloud, maskA, maskB):
        intersection_mask = maskA & maskB
        pc_x = np.reshape(np.array(getattr(pointcloud, 'x')), (-1,1))
        pc_y = np.reshape(np.array(getattr(pointcloud, 'y')), (-1,1))

        pc_A = np.concatenate((pc_x[maskA], pc_y[maskA]), axis=1)
        pc_B = np.concatenate((pc_x[maskB], pc_y[maskB]), axis=1)

        intersection = np.concatenate((pc_x[intersection_mask], pc_y[intersection_mask]), axis=1)        
        # print("pcx shape: ", pc_x.shape)
        # print("pcy shape: ", pc_y.shape)
        # print("pcA shape: ", pc_A.shape)
        # print("pcB shape: ", pc_B.shape)
        # print("intersection shape: ", intersection.shape)
        intersection_transformed = Pipeline.transform_with_pca(intersection)
        

        # cut
        mask_pos = intersection_transformed[:,1] > 0
        mask_pos_full = np.zeros((len(intersection_mask)))
        small_pos = 0
        for i in range(len(intersection_mask)):
            if intersection_mask[i]:
                mask_pos_full[i] = mask_pos[small_pos]
                small_pos += 1
        mask_neg_full = mask_pos_full == False
        # mask_neg = mask_pos == False
        # print("mask_pos shape: ", mask_pos.shape)
        # print("mask_neg shape: ", mask_neg.shape)
        # print("mask_pos_full shape: ", mask_pos_full.shape)
        # print("mask_neg_full shape: ", mask_neg_full.shape)

        # find centroids of the two clusters:
        centroid_A = np.mean(pc_A, axis=0)
        centroid_B = np.mean(pc_B, axis=0)

        centroid_pos = np.mean(intersection[mask_pos], axis=0)

        dist_pos_A = ((centroid_A[0] - centroid_pos[0])**2 + (centroid_A[1] - centroid_pos[1])**2)**0.5
        dist_pos_B = ((centroid_B[0] - centroid_pos[0])**2 + (centroid_B[1] - centroid_pos[1])**2)**0.5

        if dist_pos_A < dist_pos_B:
            maskA = (maskA.astype(bool) | mask_pos_full.astype(bool))
            maskB = (maskB.astype(bool) | mask_neg_full.astype(bool))
        else:
            maskA = (maskA.astype(bool) | mask_neg_full.astype(bool))
            maskB = (maskB.astype(bool) | mask_pos_full.astype(bool))
        
        return maskA, maskB

    def run_subprocess(self, src_script, script_name, add_to_log=True, params=None, verbose=True):
        # go at the root of the segmenter
        old_current_dir = os.getcwd()
        os.chdir(src_script)

        # add title to log
        if add_to_log:
            self.log += f"\n========\nSCRIPT: {script_name}\n========\n"

        # construct command and run subprocess
        # script_str = script_name
        script_str = ['bash', script_name]
        if params:
            for x in params:
                script_str.append(str(x))
            # script_str += ' ' + ' '.join([str(x) for x in params])
        process = subprocess.Popen(
            script_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            encoding='utf-8',
            errors='replace'
        )

        while True:
            realtime_output = process.stdout.readline()

            if realtime_output == '' and process.poll() is not None:
                break

            if realtime_output and verbose:
                print(realtime_output.strip(), flush=True)
            if add_to_log:
                self.log += realtime_output.strip() + "\n"

        # Ensure the process has fully exited
        process.wait()
        if verbose:
            print(f"[INFO] Process completed with exit code {process.returncode}")

        # go back to original working dir
        os.chdir(old_current_dir)

        # return exit code
        return process.returncode

    def preprocess(self, verbose=True):
        # remove duplicates
        for tile in self.tiles_to_process:
            tile_path = os.path.join(self.result_pseudo_labels_dir, tile)
            tile = laspy.read(tile_path)
            if verbose:
                print("size of original file: ", len(tile))

            Pipeline.remove_duplicates(tile)

            if verbose:
                print("size after duplicate removal: ", len(tile))


    def segment(self, verbose=False):
        print("Starting inference:")
        if not os.path.exists(self.preds_src):
            os.mkdir(self.preds_src)

        # select checkpoint
        if self.model_checkpoint_src == None:
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
        temp_seg_src = os.path.join(self.root_src, self.data_src, 'temp_seg')
        if os.path.exists(temp_seg_src):
            shutil.rmtree(temp_seg_src)
        os.mkdir(temp_seg_src)



        # list_files = [x for x in os.listdir(os.path.join(self.root_src, self.data_src)) if x.endswith(self.file_format) and x not in self.problematic_tiles]
        # if self.classification.processes.do_remove_empty_tiles:
        #     list_files = [x for x in list_files if x not in self.empty_tiles]

        # for _, file in tqdm(enumerate(list_files), total=len(list_files), desc="Processing"):



        # creates pack of samples to infer on
        if self.inference.num_tiles_per_inference > 1:
            list_pack_of_tiles = [self.tiles_to_process[x:min(y,len(self.tiles_to_process))] for x, y in zip(
                range(0, len(self.tiles_to_process) - self.inference.num_tiles_per_inference, self.inference.num_tiles_per_inference),
                range(self.inference.num_tiles_per_inference, len(self.tiles_to_process), self.inference.num_tiles_per_inference),
                )]
            if list_pack_of_tiles[-1][-1] != self.tiles_to_process[-1]:
                list_pack_of_tiles.append(self.tiles_to_process[(len(list_pack_of_tiles)*self.inference.num_tiles_per_inference)::])
        else:
            list_pack_of_tiles = [[x] for x in self.tiles_to_process]
        # for _, pack in tqdm(enumerate(list_pack_of_tiles), total=len(list_pack_of_tiles), desc="Processing"):
        #     if verbose:
        #         print("===\tProcessing files: ")
        #         for file in pack:
        #             print("\t", file)
        #         print("===")

        for _, pack in tqdm(enumerate(list_pack_of_tiles), total=len(list_pack_of_tiles), desc="Processing"):
            if verbose:
                print("===\tProcessing files: ")
                for file in pack:
                    print("\t", file)
                print("===")

            # create / reset temp folder
            if os.path.exists(temp_seg_src):
                shutil.rmtree(temp_seg_src)
            os.mkdir(temp_seg_src)
                
            # copy files to temp folder
            for file in pack:
                original_file_src = os.path.join(self.result_pseudo_labels_dir, self.data_src, file)
                # original_file_src = os.path.join(self.data_dest, file)
                temp_file_src = os.path.join(temp_seg_src, file)
                # print(temp_file_src)
                shutil.copyfile(original_file_src, temp_file_src)

            return_code = self.run_subprocess(
                src_script=self.segmenter_root_src,
                script_name="./run_oracle_pipeline.sh",
                params= [temp_seg_src, temp_seg_src],
                verbose=verbose
                )

            # catch errors
            if return_code != 0:
                if verbose:
                    print(f"Problem with tiles:")
                    for file in pack:
                        print("\t", file)
                for file in pack:
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

                # # check if empty
                # src_pred_file = os.path.join(self.preds_src, file.split('.')[0] + '_out.' + self.file_format)
                # tile = laspy.read(src_pred_file)
                # num_instances = len(set(tile.PredInstance))
                # if num_instances == 1:
                #     if verbose:
                #         print(f"Empty tile: {file}")
                #     self.empty_tiles.append(file)
                #     if self.classification.processes.do_remove_empty_tiles:
                #         os.remove(os.path.join(src_pred_file))
                #         # if file in self.tiles_to_process:
                #         #     self.tiles_to_process.remove(file)
                if verbose:
                    print("Segmentation done!")
        # # loops on samples:
        # for _, file in tqdm(enumerate(self.tiles_to_process), total=len(self.tiles_to_process), desc="Processing"):
        #     if verbose:
        #         print("===\tProcessing file: ", file, "\t===")
        #     # original_file_src = os.path.join(os.path.join(self.root_src, self.data_src, file))
        #     original_file_src = os.path.join(self.result_pseudo_labels_dir, self.data_src, file)
        #     temp_file_src = os.path.join(os.path.join(temp_seg_src, file))
        #     shutil.copyfile(
        #         original_file_src,
        #         temp_file_src,
        #     )
            
        #     return_code = self.run_subprocess(
        #         src_script=self.segmenter_root_src,
        #         script_name="./run_oracle_pipeline.sh",
        #         params= [temp_seg_src, temp_seg_src],
        #         verbose=verbose
        #         )
        
        #     # catch errors
        #     if return_code != 0:
        #         if verbose:
        #             print(f"Problem with tile {file}")
        #         self.problematic_tiles.append(file)
        #     else:
        #         # unzip results
        #         if verbose:
        #             print("Unzipping results...")
        #         self.unzip_laz_files(
        #             zip_path=os.path.join(temp_seg_src, "results.zip"),
        #             extract_to=self.preds_src,
        #             delete_zip=True
        #             )

        #         # check if empty
        #         src_pred_file = os.path.join(self.preds_src, file.split('.')[0] + '_out.' + self.file_format)
        #         tile = laspy.read(src_pred_file)
        #         num_instances = len(set(tile.PredInstance))
        #         if num_instances == 1:
        #             if verbose:
        #                 print(f"Empty tile: {file}")
        #             self.empty_tiles.append(file)
        #             if self.classification.processes.do_remove_empty_tiles:
        #                 os.remove(os.path.join(src_pred_file))
        #                 # if file in self.tiles_to_process:
        #                 #     self.tiles_to_process.remove(file)
        #         if verbose:
        #             print("Segmentation done!")

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
            split_instance(os.path.join(self.preds_src, file), verbose=verbose)

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
            # convert_all_in_folder(src_folder_in=output_folder, src_folder_out=output_folder, in_type='pcd', out_type='laz')
            self.run_subprocess(
                src_script='/home/pdm',
                script_name="run_format_conversion.sh",
                params=[output_folder, output_folder, 'pcd', 'laz'],
                verbose=verbose
            )

            # remove pcd files
            for file in os.listdir(output_folder):
                if file.endswith('.pcd'):
                    os.remove(os.path.join(output_folder, file))
            
    def create_pseudo_labels(self, verbose=False):
        print("Creating pseudo labels:")
        self.log += "Creating pseudo labels:\n"
        list_folders = [x for x in os.listdir(self.preds_src) if os.path.abspath(os.path.join(self.preds_src, x)) and x.endswith('instance')]
        for _, child in tqdm(enumerate(list_folders), total=len(list_folders), desc='Processing'):
            if verbose:
                print("Processing sample : ", child)
            
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
            Pipeline.remove_duplicates(original_file)
            Pipeline.remove_duplicates(new_file)
            Pipeline.remove_duplicates(pred_file)
            Pipeline.match_pointclouds(new_file, pred_file)

            coords_A = np.stack((original_file.x, original_file.y, original_file.z), axis=1)
            coords_original_file_view = coords_A.view([('', coords_A.dtype)] * 3).reshape(-1)

            # initialisation if first loop
            if self.current_loop == 0:
                # add pseudo-labels attribute if non-existant
                new_file.add_extra_dim(
                    laspy.ExtraBytesParams(
                        name='treeID',
                        type="f4",
                        description='Instance p-label'
                    )
                )
            
                # reset values of pseudo-labels
                new_file.classification = np.zeros(len(new_file), dtype="f4")
                new_file.treeID = np.zeros(len(new_file), dtype="f4")

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
                # if row['class'] in [1, 2]:
                if row['class'] == 2:
                    self.classified_clusters.append((coords_B_view, row['class']))

            # create masks on the original tile for each cluster (multiprocessing)
            with ProcessPoolExecutor() as executor:
                partialFunc = partial(self.process_row, coords_original_file_view)
                results = list(tqdm(executor.map(partialFunc, range(len(self.classified_clusters))), total=len(self.classified_clusters), smoothing=0.9, desc="Updating pseudo-label", disable=~verbose))
            
            # Update the original file based on results and update the csv file with ref to trees
            dict_reasons_of_rejection = {x:0 for x in ['total', 'is_ground', 'same_heighest_point', 'overlapping_greater_than_2', 'i_o_new_tree_greater_than_70_per', 'splitting_using_pca']}
            # print(dict_reasons_of_rejection)
            # print(pd.DataFrame(index=dict_reasons_of_rejection.keys(), data=dict_reasons_of_rejection.values()))
            # quit()
            id_new_tree = len(set(new_file.treeID))
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
                # new_file.treeID[mask] = id_tree
                
                # check if new tree
                corresponding_instances = new_file.treeID[mask]
                # corresponding_semantic = new_file.classification[mask]
                is_new_tree = False
                if verbose:
                    print("Set of overlapping instances: ", set(corresponding_instances))
                self.log += f"Set of overlapping instances: {set(corresponding_instances)}\n"
                
                if len(set(corresponding_instances)) == 1 and corresponding_instances[0] == 0:
                    is_new_tree = True
                    
                    if verbose:
                        print(f"Adding treeID {id_new_tree} because only grey and ground: ")
                    self.log += f"Adding treeID {id_new_tree} because only grey and ground: \n"
                else:
                    if verbose:
                        print(f"Comparing to existing values")
                    self.log += f"Comparing to existing values \n"

                    is_new_tree = True
                    for instance in set(corresponding_instances):
                        # if ground
                        if instance == 0:
                            dict_reasons_of_rejection["is_ground"] += 1
                            is_new_tree = False
                            # break

                        other_tree_mask = new_file.treeID == instance
                        new_file_x = np.array(getattr(new_file, 'x')).reshape((-1,1))
                        new_file_y = np.array(getattr(new_file, 'y')).reshape((-1,1))
                        new_file_z = np.array(getattr(new_file, 'z')).reshape((-1,1))
 
                        # compare heighest points
                        if np.max(new_file_z[mask]) == np.max(new_file_z[other_tree_mask]):
                            dict_reasons_of_rejection["same_heighest_point"] += 1
                            is_new_tree = False
                            # break
                        
                        # get intersection
                        intersection_mask = mask & other_tree_mask
                        # intersection = np.vstack((new_file_x[intersection_mask], new_file_y[intersection_mask], new_file_z[intersection_mask]))
                        intersection = np.concatenate((new_file_x, new_file_y), axis=1)[intersection_mask]
                        if verbose:
                            print(f"Comparing to existing tree with id {instance} of size {np.sum(other_tree_mask)} and intersection of size {np.sum(intersection_mask)}")
                        self.log += f"Comparing to existing tree with id {instance} of size {np.sum(other_tree_mask)} and intersection of size {np.sum(intersection_mask)} \n"

                        # check radius of intersection
                        intersection_pca = Pipeline.transform_with_pca(intersection)
                        small_range = np.max(intersection_pca[:,1]) - np.min(intersection_pca[:,1])
                        if small_range > 2:
                        # range_x = np.min(intersection[0,:]) - np.max(intersection[0,:])
                        # range_y = np.min(intersection[1,:]) - np.max(intersection[1,:])
                        # range_z = np.min(intersection[2,:]) - np.max(intersection[2,:])
                        # if range_x > 4 or range_y > 4 or range_z > 4:
                            is_new_tree = False
                            dict_reasons_of_rejection["overlapping_greater_than_2"] += 1
                            # break

                        # intersection over new tree
                        if np.sum(intersection_mask) / np.sum(mask) > 0.7:
                            is_new_tree = False
                            dict_reasons_of_rejection["i_o_new_tree_greater_than_70_per"] += 1
                            # break 
                            
                        if is_new_tree == False:
                            dict_reasons_of_rejection["total"] += 1

                if is_new_tree == True:
                    # update classification
                    new_file.classification[mask] = 4

                    # update instances
                    if verbose:
                        print("Length of corresponding instances: ", len(set(corresponding_instances)))
                    if len(set(corresponding_instances)) > 1:
                        for instance in set(corresponding_instances):
                            if instance == 0 or len:
                                continue
                            dict_reasons_of_rejection["splitting_using_pca"] += 1
                            other_tree_mask = new_file.treeID == instance
                            intersection_mask = mask & other_tree_mask

                            if len(intersection_mask) > 1:
                                mask, new_other_tree_mask = Pipeline.split_instances(new_file, mask, other_tree_mask)
                        
                            # test if trees are still recognisable
                            # later...

                    new_file.treeID[mask] = id_new_tree
                    id_new_tree += 1

                    if verbose:
                        print("New tree with instance: ", id_new_tree)
                    self.log += f"New tree with instance: {id_new_tree} \n"

                
                # set semantic p-label
                
                # test if 


                # if value == 1 or (self.garbage_as_grey and value == 0):
                #     new_file.classification[mask] = 0
                # elif value == 0 and not self.garbage_as_grey:
                #     new_file.classification[mask] = 1
                # elif value == 2:
                #     new_file.classification[mask] = 4
                # else:
                #     raise ValueError("Problem in the setting of the semantic pseudo-labels!")
                
            # saving back original file and also to corresponding loop if flag set to
            new_file.write(original_file_src)
            pd.DataFrame(index=dict_reasons_of_rejection.keys(), data=dict_reasons_of_rejection.values(), columns=['count']).to_csv(os.path.join(self.result_current_loop_dir, 'pseudo_labels_reasons_of_reject.csv'), sep=';')
            if self.save_pseudo_labels_per_loop:
                os.makedirs(os.path.join(self.result_current_loop_dir, "pseudo_labels"), exist_ok=True)
                loop_file_src = os.path.join(self.result_current_loop_dir, "pseudo_labels", os.path.basename(original_file_src))
                new_file.write(loop_file_src)
            
        # removing unsused tiles from pseudo-label directory (so that it is not processed by the training phase)
        list_tiles_pseudo_labels = [x for x in os.listdir(self.result_pseudo_labels_dir) if x.endswith(self.file_format)]
        for tile_name in list_tiles_pseudo_labels:
            if tile_name not in self.tiles_to_process:
                os.remove(os.path.join(self.result_pseudo_labels_dir, tile_name))

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

    def prepare_data(self, verbose=True):
        print("Prepare data:")
        self.run_subprocess(
            src_script="/home/pdm/models/SegmentAnyTree/",
            script_name="./run_sample_data_conversion.sh",
            # params= [self.data_src],
            params= [self.result_pseudo_labels_dir],
            verbose=verbose
            )

    def train(self, verbose=True):
        # test if enough tiles available
        for type in ['train', 'test', 'val']:
            if len([x for x in os.listdir(os.path.join(self.result_pseudo_labels_dir, 'treeinsfused/raw/pseudo_labels')) if x.split('.')[0].endswith(type)]) == 0:
                raise InterruptedError(f"No {type} tilse for training process!!!")

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
            val="../../" + os.path.normpath(self.results_root_src) + "/" + self.result_src_name + '/' + str(self.current_loop),
            # val=self.result_current_loop_dir,
        )

        # run training script
        model_checkpoint = self.model_checkpoint_src if self.model_checkpoint_src != None else "/home/pdm/models/SegmentAnyTree/model_file"
        # print(self.model_checkpoint_src)
        # print(model_checkpoint)
        self.run_subprocess(
            src_script="/home/pdm/models/SegmentAnyTree/",
            script_name="./run_pipeline.sh",
            params= [self.training.num_epochs_per_loop, 
                     self.training.batch_size, 
                     self.training.sample_per_epoch,
                     model_checkpoint,
                     self.training_metrics_src,
                     self.current_loop,
                     ],
            verbose=verbose,
            )
        
        # update path to checkpoint
        # self.model_checkpoint_src = os.path.join(self.training.result_training_dir, str(self.current_loop))
        # self.model_checkpoint_src = self.result_current_loop_dir

    def visualization(self):
        print("Saving metrics visualizations")
        
        # creates location
        location_src = os.path.join(self.result_dir, "images")
        os.makedirs(location_src, exist_ok=True)
        
        show_global_metrics(
            src_data=self.training_metrics_src,
            src_location=os.path.join(location_src, 'training_metrics.png'),
            show_figure=False,
            save_figure=True,
            )
        show_inference_counts(
            self.inference_metrics_src,
            src_location=os.path.join(location_src, 'inference_count.png'),
            show_figure=False,
            save_figure=True,
            )
        show_inference_metrics(
            self.inference_metrics_src,
            src_location=os.path.join(location_src, 'inference_metrics.png'),
            show_figure=False,
            save_figure=True,
            )

    def test(self):
        # test if enough tiles available
        # for type in ['train', 'test', 'val']:
        #     if len([x for x in os.listdir(os.path.join(self.result_pseudo_labels_dir, 'treeinsfused/raw/pseudo_labels')) if x.split('.')[0].endswith(type)]) == 0:
        #         raise InterruptedError(f"No {type} tilse for training process!!!")

        print("Training:")
        # modify dataset config file
        # Pipeline.change_var_val_yaml(
        #     src_yaml=self.training.config_data_src,
        #     var='data/dataroot',
        #     # val=os.path.join(self.result_pseudo_labels_dir),
        #     val=os.path.join(self.result_pseudo_labels_dir),
        # )

        # # modify results directory
        # Pipeline.change_var_val_yaml(
        #     src_yaml=self.training.config_results_src,
        #     var='hydra/run/dir',
        #     # val="../../" + os.path.normpath(self.results_root_src) + "/" + self.result_src_name + '/' + str(self.current_loop),
        #     val=self.result_current_loop_dir,
        # )

        # run training script
        # model_checkpoint = self.model_checkpoint_src if self.model_checkpoint_src != "None" else "/home/pdm/models/SegmentAnyTree/model_file"
        model_checkpoint = "/home/pdm/models/SegmentAnyTree/model_file"
        self.run_subprocess(
            src_script="/home/pdm/models/SegmentAnyTree/",
            script_name="./run_pipeline.sh",
            params= [self.training.num_epochs_per_loop, 
                     self.training.batch_size, 
                     self.training.sample_per_epoch,
                     model_checkpoint,
                     self.training_metrics_src,
                     self.current_loop,
                     ],
            )

    def save_log(self, dest, clear_after=True, verbose=False):
        if verbose:
            print(f"Saving logs (of size {len(self.log)}) to : {dest}")
        
        os.makedirs(dest, exist_ok=True)

        with open(os.path.join(dest, "log.txt"), "w") as file:
            file.write(self.log)
        if clear_after:
            self.log = ""

if __name__ == "__main__":




    # a = [1, 2, 3]
    # b = [4, 5, 6]
    # c = np.vstack((a,b))
    # print(c.shape)
    # quit()

    from time import time
    import torch

    cfg_dataset = OmegaConf.load('./config/dataset.yaml')
    cfg_preprocess = OmegaConf.load('./config/preprocessing.yaml')
    cfg_pipeline = OmegaConf.load('./config/pipeline.yaml')
    cfg_classifier = OmegaConf.load('./config/classifier.yaml')
    cfg_segmenter = OmegaConf.load('./config/segmenter.yaml')
    cfg = OmegaConf.merge(cfg_dataset, cfg_preprocess, cfg_pipeline, cfg_classifier, cfg_segmenter)

    # load pipepline arguments
    ROOT_SRC = cfg.pipeline.root_src

    # load data
    NUM_LOOPS = cfg.pipeline.num_loops
    DATA_SRC = os.path.join(cfg.dataset.project_root_src, cfg.dataset.data_src)
    FILE_FORMAT = cfg.dataset.file_format

    TRAIN_FRAC = cfg.pipeline.train_frac
    TEST_FRAC = cfg.pipeline.test_frac
    VAL_FRAC = cfg.pipeline.val_frac

    # processes
    # SAVE_PSEUDO_LABELS_PER_LOOP = cfg.pipeline.processes.save_pseudo_labels_per_loop


    # assertions
    assert TRAIN_FRAC + TEST_FRAC + VAL_FRAC == 1.0

    # start timer
    time_start_process = time()

    # create pipeline
    # pipeline = Pipeline(cfg) 
    # pipeline.result_pseudo_labels_dir = r"D:\PDM_repo\Github\PDM\results\trainings\20250505_134754_test\pseudo_labels"
    # pipeline.data_src = os.path.join(pipeline.data_src, "loops/0")
    # pipeline.preds_src = os.path.join(pipeline.data_src, "preds")
    # pipeline.create_pseudo_labels()
    path = r"D:\PDM_repo\Github\PDM\results\trainings\20250505_151920_test\pseudo_labels\treeinsfused\processed_0.2\train.pt"
    data, slices = torch.load(path)
    print(data)
    print(slice)