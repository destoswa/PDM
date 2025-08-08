import os
import sys
import shutil
import zipfile
import numpy as np
import pandas as pd
import laspy
import subprocess
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
if __name__ == "__main__":
    sys.path.append(os.getcwd())
from src.format_conversions import convert_all_in_folder
from src.metrics import compute_classification_results, compute_panoptic_quality
from src.visualization import show_global_metrics, show_inference_counts, show_inference_metrics, show_pseudo_labels_evolution, show_pseudo_labels_vs_gt, show_training_losses, show_stages_losses, show_test_set
from src.splitting import split_instance
from src.fast_inference import fast_inference


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
        self.save_pseudo_labels_per_loop = cfg.pipeline.processes.save_pseudo_labels_per_loop
        self.do_continue_from_existing = cfg.pipeline.preload.do_continue_from_existing
        self.log = ""

        # config regarding preprocess
        self.do_flatten = cfg.pipeline.processes.do_flatten
        self.flatten_tile_size = cfg.pipeline.processes.flatten_tile_size

        # config regarding inference
        self.inference = cfg.segmenter.inference
        self.preds_src = ""
        self.problematic_tiles = []
        self.empty_tiles = []

        # config regarding classification
        self.classification = cfg.classifier
        self.src_preds_classifier = None
        self.classified_clusters = None

        # config regarding pseudo-labels
        self.do_add_multi_as_trees_semantic = cfg.pipeline.processes.do_add_multi_as_trees_semantic

        # config regarding training
        self.training = cfg.segmenter.training
        self.model_checkpoint_src = None
        self.current_epoch = 0

        # config regarding results
        self.result_src_name_suffixe = cfg.pipeline.result_src_name_suffixe
        self.result_src_name = datetime.now().strftime(r"%Y%m%d_%H%M%S_") + self.result_src_name_suffixe

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


        #   _remove data processes if necessary
        if not cfg.pipeline.debugging.keep_previous_data and not cfg.pipeline.preload.do_continue_from_existing and os.path.exists(os.path.join(self.data_src, 'loops')):
            shutil.rmtree(os.path.join(self.data_src, 'loops'))

        # update model to use if starting from existing pipeline
        if self.do_continue_from_existing:
            self.model_checkpoint_src = os.path.join(self.result_dir, str(self.current_loop - 1))

        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.result_pseudo_labels_dir, exist_ok=True)
        os.makedirs(self.result_current_loop_dir, exist_ok=True)

        #   _add copy of cfg in results
        with open(os.path.join(self.result_dir, "configuration.yaml"), 'w+') as file:
            OmegaConf.save(config=cfg, f=file.name)

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
            input_data_loc = os.path.join(self.data_src, 'originals') if self.do_flatten else self.data_src
            for _, file in tqdm(enumerate(self.tiles_to_process), total=len(self.tiles_to_process), desc="Process"):
                shutil.copyfile(
                    os.path.join(input_data_loc, file),
                    os.path.join(self.result_pseudo_labels_dir, file)
                )
                
            self.training_metrics.to_csv(self.training_metrics_src, sep=';', index=False)
            self.inference_metrics.to_csv(self.inference_metrics_src, sep=';', index=False)
        else:
            # test if the training file already exists
            if os.path.exists(self.training_metrics_src):
                self.training_metrics = pd.read_csv(self.training_metrics_src, sep=';')
                self.training_metrics = self.training_metrics[self.training_metrics.num_loop < self.current_loop]
            else:
                self.training_metrics.to_csv(self.training_metrics_src, sep=';', index=False)

            # test if the inference file already exists
            if os.path.exists(self.inference_metrics_src):
                self.inference_metrics = pd.read_csv(self.inference_metrics_src, sep=';')
                self.inference_metrics = self.inference_metrics[self.inference_metrics.num_loop < self.current_loop]
            else:
                self.inference_metrics.to_csv(self.inference_metrics_src, sep=';', index=False)

        # testing
        assert len([x for x in os.listdir(os.path.join(self.root_src, self.data_src)) if x.endswith(self.file_format)]) > 0     # dataset is not empty

    @staticmethod
    def unzip_laz_files(zip_path, extract_to=".", delete_zip=True):
        """
        Extract all .laz files from a zip archive to a specified directory.

        Args:
            - zip_path (str): Path to the .zip archive containing .laz files.
            - extract_to (str, optional): Directory to extract the files to. Defaults to current directory.
            - delete_zip (bool, optional): Whether to delete the zip file after extraction. Defaults to True.

        Returns:
            - None: Extracts files and optionally deletes the zip archive.
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
        """
        Modify a variable value in a YAML file.

        Args:
            - src_yaml (str): Path to the YAML file.
            - var (str): Path to the variable in the YAML file, using '/' as separator.
            - val (Any): New value to assign to the specified variable.

        Returns:
            - None: The function modifies the YAML file in place without returning anything.
        """

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
    def remove_duplicates(laz_file, decimals=2):
        """
        Remove duplicate points from a LAS/LAZ file based on rounded coordinate precision.

        Args:
            - laz_file (laspy.LasData): Input LAS/LAZ file as a laspy object.
            - decimals (int, optional): Number of decimal places to round coordinates for duplicate detection. Defaults to 2.

        Returns:
            - laspy.LasData: A new LAS/LAZ file object with duplicates removed.
        """

        coords = np.round(np.vstack((laz_file.x, laz_file.y, laz_file.z)).T, decimals)
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        mask = np.zeros(len(coords), dtype=bool)
        mask[unique_indices] = True

        # Create new LAS object
        header = laspy.LasHeader(point_format=laz_file.header.point_format, version=laz_file.header.version)
        new_las = laspy.LasData(header)

        setattr(new_las, 'x', np.array(laz_file.x)[mask])
        setattr(new_las, 'y', np.array(laz_file.y)[mask])
        setattr(new_las, 'z', np.array(laz_file.z)[mask])
        for dim in [x for x in laz_file.point_format.dimension_names if x not in ['X', 'Y', 'Z']]:
            setattr(new_las, dim, np.array(laz_file[dim])[mask])

        return new_las

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
        """
        Transform a 2D or 3D point cloud using Principal Component Analysis (PCA) to align with principal axes.

        Args:
            - pointcloud (np.ndarray): Input point cloud as a numpy array of shape (N, 2) or (N, 3).
            - verbose (bool, optional): Whether to print PCA components and transformed points. Defaults to False.

        Returns:
            - np.ndarray: PCA-transformed point cloud of shape (N, 2).
        """

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
        """
        Split overlapping instances within a point cloud using PCA to determine a separation line.

        Args:
            - pointcloud (laspy.LasData): Input LAS/LAZ point cloud.
            - maskA (np.ndarray): Boolean mask for instance A.
            - maskB (np.ndarray): Boolean mask for instance B.

        Returns:
            - tuple[np.ndarray, np.ndarray]: Updated boolean masks (maskA, maskB) after splitting the intersection region based on proximity to instance centroids.
        """

        intersection_mask = maskA & maskB
        pc_x = np.reshape(np.array(getattr(pointcloud, 'x')), (-1,1))
        pc_y = np.reshape(np.array(getattr(pointcloud, 'y')), (-1,1))

        pc_A = np.concatenate((pc_x[maskA], pc_y[maskA]), axis=1)
        pc_B = np.concatenate((pc_x[maskB], pc_y[maskB]), axis=1)

        intersection = np.concatenate((pc_x[intersection_mask], pc_y[intersection_mask]), axis=1)        
        intersection_transformed = Pipeline.transform_with_pca(intersection)

        # cut
        mask_pos = intersection_transformed[:,1] > 0
        mask_neg = mask_pos == False
        mask_pos_full = np.zeros((len(intersection_mask)))
        mask_neg_full = np.zeros((len(intersection_mask)))
        small_pos = 0
        small_neg = 0
        for i in range(len(intersection_mask)):
            if intersection_mask[i]:
                mask_pos_full[i] = mask_pos[small_pos]
                small_pos += 1
        for i in range(len(intersection_mask)):
            if intersection_mask[i]:
                mask_neg_full[i] = mask_neg[small_neg]
                small_neg += 1

        # find centroids of the two clusters:
        centroid_A = np.mean(pc_A, axis=0)
        centroid_B = np.mean(pc_B, axis=0)

        centroid_pos = np.mean(intersection[mask_pos], axis=0)

        dist_pos_A = ((centroid_A[0] - centroid_pos[0])**2 + (centroid_A[1] - centroid_pos[1])**2)**0.5
        dist_pos_B = ((centroid_B[0] - centroid_pos[0])**2 + (centroid_B[1] - centroid_pos[1])**2)**0.5

        # remove intersection from masks
        anti_intersection_mask = intersection_mask == False
        maskA = maskA.astype(bool) & anti_intersection_mask.astype(bool)
        maskB = maskB.astype(bool) & anti_intersection_mask.astype(bool)

        # add part of intersection to each mask
        if dist_pos_A < dist_pos_B:
            maskA = (maskA.astype(bool) | mask_pos_full.astype(bool))
            maskB = (maskB.astype(bool) | mask_neg_full.astype(bool))
        else:
            maskA = (maskA.astype(bool) | mask_neg_full.astype(bool))
            maskB = (maskB.astype(bool) | mask_pos_full.astype(bool))
        
        return maskA, maskB

    def run_subprocess(self, src_script, script_name, add_to_log=True, params=None, verbose=True):
        """
        Run a subprocess in a specific directory with optional parameters.

        Args:
            - src_script (str): Directory path where the subprocess will be executed.
            - script_name (str): Name of the script to run (bash file).
            - params (list, optional): List of parameters to pass to the script. Defaults to None.
            - verbose (bool, optional): Whether to print real-time subprocess output. Defaults to True.

        Returns:
            - int: Exit code returned by the subprocess.
        """
        # go at the root of the segmenter
        old_current_dir = os.getcwd()
        os.chdir(src_script)

        # add title to log
        if add_to_log:
            self.log += f"\n========\nSCRIPT: {script_name}\n========\n"

        # construct command and run subprocess
        script_str = ['bash', script_name]
        if params:
            for x in params:
                script_str.append(str(x))
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
        """
        Preprocess point cloud tiles by removing duplicate points before further processing.

        Args:
            - verbose (bool, optional): Whether to print the size of files before and after duplicate removal. Defaults to True.

        Returns:
            - None: Overwrites the original tiles with duplicate-free versions.
        """

        # remove duplicates
        for tile in self.tiles_to_process:
            tile_path = os.path.join(self.result_pseudo_labels_dir, tile)
            tile = laspy.read(tile_path)
            if verbose:
                print("size of original file: ", len(tile))

            tile = Pipeline.remove_duplicates(tile)
            tile.write(tile_path)

            if verbose:
                print("size after duplicate removal: ", len(tile))

    def segment(self, verbose=False):
        """
        Segment point cloud tiles by running an external segmentation pipeline, handling optional flattening and faulty tiles.

        Args:
            - verbose (bool, optional): Whether to print detailed progress information during segmentation. Defaults to False.

        Returns:
            - None: Runs segmentation, updates prediction files, and tracks problematic or empty tiles.
        """

        print("Starting inference:")
        os.makedirs(self.preds_src, exist_ok=True)

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

        # if flattening
        original_tiles_to_process = []
        original_tiles_src = os.path.dirname(self.data_src)
        original_preds_src = self.preds_src
        if self.do_flatten:
            self.preds_src = self.preds_src + "_flatten"
            os.makedirs(self.preds_src, exist_ok=True)
            self.data_src = os.path.join(self.data_src, 'flatten')
            new_tiles_to_process = []
            for tile in self.tiles_to_process:
                for tile_new in os.listdir(self.data_src):
                    if tile_new.split('_flatten')[0] == tile.split('.' + self.file_format)[0]:
                        new_tiles_to_process.append(tile_new)
            if len(self.tiles_to_process) != len(new_tiles_to_process):
                raise ValueError("Could not find all the corresponding flatten tiles to the original ones")
            original_tiles_to_process = self.tiles_to_process
            self.tiles_to_process = new_tiles_to_process

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
                temp_file_src = os.path.join(temp_seg_src, file)
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

                if verbose:
                    print("Segmentation done!")

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

        
        if self.do_flatten:
            # reorder predictions, transfert preds to originals and replace flatten with originals
            for id_tile, tile_flatten in enumerate(self.tiles_to_process):
                laz_flatten = laspy.read(os.path.join(self.data_src, tile_flatten))
                laz_preds = laspy.read(os.path.join(self.preds_src, tile_flatten.split(".laz")[0] +"_out.laz"))
                laz_original = laspy.read(os.path.join(original_tiles_src, original_tiles_to_process[id_tile]))
                self.match_pointclouds(laz_flatten, laz_preds)
                for new_col_name in ["PredInstance", "PredSemantic"]:
                    if new_col_name not in list(laz_original.point_format.dimension_names):
                        new_col = laspy.ExtraBytesParams(name=new_col_name, type=np.uint16)
                        laz_original.add_extra_dim(new_col)
                    laz_original[new_col_name] = laz_preds[new_col_name]

                laz_original.write(os.path.join(original_preds_src, original_tiles_to_process[id_tile].split('.laz')[0] + '_out.laz'))

            self.data_src = self.data_src.split('/flatten')[0]
            self.tiles_to_process = original_tiles_to_process

            shutil.copytree(self.preds_src, os.path.join(self.result_current_loop_dir, os.path.basename(self.preds_src)))
            self.preds_src = original_preds_src
            
        # copy preds to results
        shutil.copytree(self.preds_src, os.path.join(self.result_current_loop_dir, os.path.basename(self.preds_src)))
    
    def classify(self, verbose=False):
        """
        Classify segmented point cloud instances by splitting instances, converting formats, running inference, and post-processing.

        Args:
            - verbose (bool, optional): Whether to display progress and warnings during classification. Defaults to False.

        Returns:
            - None: Performs classification on predicted segments and stores results in the appropriate format.
        """

        print("Starting classification:")

        # loop on files:
        list_files = [f for f in os.listdir(self.preds_src) if f.endswith(self.file_format)]
        for _, file in tqdm(enumerate(list_files), total=len(list_files), desc="Processing"):
            split_instance(os.path.join(self.preds_src, file), verbose=verbose)
            dir_target = self.preds_src + '/' + file.split('/')[-1].split('.')[0] + "_split_instance"

            # verify that no file is corrupted
            for cluster in [x for x in os.listdir(dir_target) if x.endswith(self.file_format)]:
                try:
                    laspy.read(os.path.join(dir_target, cluster))
                except Exception as e:
                    if verbose:
                        print(f"Following cluster is corrupted and removed from processing: {os.path.join(dir_target, cluster)}")
                    self.log += f"Following cluster is corrupted and removed from processing: {os.path.join(dir_target, cluster)} \n"
                    os.makedirs(os.path.join(dir_target, 'corrupted_files'), exist_ok=True)
                    os.rename(
                        os.path.join(dir_target, cluster),
                        os.path.join(dir_target, 'corrupted_files', cluster)
                    )

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
        """
        Create and update pseudo labels for segmented point cloud instances by matching clusters, checking overlaps, splitting instances, and applying classification rules.

        Args:
            - verbose (bool, optional): Whether to print detailed progress and logging information during processing. Defaults to False.

        Returns:
            - None: Updates the pseudo-labeled point clouds and writes the results to disk, including monitoring statistics and optional flattened versions.
        """

        print("Creating pseudo labels:")
        self.log += "Creating pseudo labels:\n"

        # load arguments of classifier
        args_classifier = OmegaConf.load("./models/KDE_classifier/config/inference.yaml")

        # loop on samples
        dict_monitoring = {x:0 for x in ['total', 'new_instance', 'is_ground', 'same_heighest_point', 'overlapping_greater_than_2', 'i_o_new_tree_greater_than_70_per', 'i_o_other_tree_greater_than_70_per', 'splitting_using_pca', 'not_a_tree_after_splitting']}
        list_folders = [x for x in os.listdir(self.preds_src) if os.path.abspath(os.path.join(self.preds_src, x)) and x.endswith('instance')]
        for _, child in tqdm(enumerate(list_folders), total=len(list_folders), desc='Processing'):
            if verbose:
                print("Processing sample : ", child)
            
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
            original_file = Pipeline.remove_duplicates(original_file)
            new_file = Pipeline.remove_duplicates(new_file)
            pred_file = Pipeline.remove_duplicates(pred_file)
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
            self.multi_clusters = []
            for _, row in df_results.iterrows():
                cluster_path = os.path.join(full_path, row.file_name.split('.pcd')[0] + '.' + self.file_format)
                if not os.path.exists(cluster_path):
                    continue
                if row['class'] == 0:
                    continue

                cluster = laspy.open(cluster_path, mode='r').read()
                coords = np.stack((cluster.x, cluster.y, cluster.z), axis=1)
                coords_view = coords.view([('', coords.dtype)] * 3).reshape(-1)
                # if row['class'] in [1, 2]:
                if row['class'] == 2:
                    self.classified_clusters.append((coords_view, row['class']))
                elif row['class'] == 1:
                    self.multi_clusters.append((coords_view, row['class']))

            # Processing multi
            if self.do_add_multi_as_trees_semantic:
                #   _Create masks on the original tile for each cluster (multiprocessing)
                results = []
                with ProcessPoolExecutor() as executor:
                    partialFunc = partial(self.process_row_multi, coords_original_file_view)
                    results = list(tqdm(executor.map(partialFunc, range(len(self.multi_clusters))), total=len(self.multi_clusters), smoothing=0.9, desc="Updating pseudo-label", disable=~verbose))

                #   _Update the original file based on results and update the csv file with ref to trees
                if len(self.multi_clusters) > 0:
                    for id_tree, (mask, value) in tqdm(enumerate(results), total=len(results), desc='temp', disable=~verbose):
                        new_file.classification[mask] = 4
    
            # Processing singles
            id_new_tree = len(set(new_file.treeID))

            #   _Create masks on the original tile for each cluster (multiprocessing)
            results = []
            with ProcessPoolExecutor() as executor:
                partialFunc = partial(self.process_row_single, coords_original_file_view)
                results = list(tqdm(executor.map(partialFunc, range(len(self.classified_clusters))), total=len(self.classified_clusters), smoothing=0.9, desc="Updating pseudo-label", disable=~verbose))

            #   _Update the original file based on results and update the csv file with ref to trees
            for id_tree, (mask, value) in tqdm(enumerate(results), total=len(results), desc='temp', disable=~verbose):
                if verbose:
                    print(f"Processing prediction {id_tree} of size {np.sum(mask)}")
                self.log += f"Processing prediction {id_tree} of size {np.sum(mask)} \n"
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

                # Check if new tree
                corresponding_instances = new_file.treeID[mask]
                is_new_tree = True
                if verbose:
                    print("Set of overlapping instances: ", set(corresponding_instances))
                self.log += f"Set of overlapping instances: {set(corresponding_instances)}\n"
                
                if len(set(corresponding_instances)) == 1 and corresponding_instances[0] == 0:                    
                    if verbose:
                        print(f"Adding treeID {id_new_tree} because only grey and ground: ")
                    self.log += f"Adding treeID {id_new_tree} because only grey and ground: \n"
                else:
                    if verbose:
                        print(f"Comparing to existing values")
                    self.log += f"Comparing to existing values \n"

                    for instance in set(corresponding_instances):
                        # Don't compare if other instance is the ground
                        if instance == 0:
                            dict_monitoring["is_ground"] += 1
                            continue

                        other_tree_mask = new_file.treeID == instance
                        new_file_x = np.array(getattr(new_file, 'x')).reshape((-1,1))
                        new_file_y = np.array(getattr(new_file, 'y')).reshape((-1,1))
                        new_file_z = np.array(getattr(new_file, 'z')).reshape((-1,1))
                        pointCloud = np.stack([new_file_x, new_file_y, new_file_z], axis=1)

 
                        # Compare heighest points
                        if np.max(new_file_z[mask]) == np.max(new_file_z[other_tree_mask]):
                            dict_monitoring["same_heighest_point"] += 1
                            is_new_tree = False
                        
                        # Get intersection
                        intersection_mask = mask & other_tree_mask
                        if np.sum(intersection_mask) > 1:
                            intersection = np.concatenate((new_file_x, new_file_y), axis=1)[intersection_mask]
                            if verbose:
                                print(f"Comparing to existing tree with id {instance} of size {np.sum(other_tree_mask)} and intersection of size {np.sum(intersection_mask)}")
                            self.log += f"Comparing to existing tree with id {instance} of size {np.sum(other_tree_mask)} and intersection of size {np.sum(intersection_mask)} \n"

                            # Check radius of intersection
                            intersection_pca = Pipeline.transform_with_pca(intersection)
                            small_range = np.max(intersection_pca[:,1]) - np.min(intersection_pca[:,1])
                            if small_range > 2:
                                is_new_tree = False
                                dict_monitoring["overlapping_greater_than_2"] += 1

                        # Intersection over new tree
                        if np.sum(intersection_mask) / np.sum(mask) > 0.7:
                            is_new_tree = False
                            dict_monitoring["i_o_new_tree_greater_than_70_per"] += 1

                        # Intersection over other tree
                        if np.sum(intersection_mask) / np.sum(other_tree_mask) > 0.7:
                            is_new_tree = False
                            dict_monitoring["i_o_other_tree_greater_than_70_per"] += 1
                            
                        if is_new_tree == False:
                            dict_monitoring["total"] += 1

                if is_new_tree == True:
                    # Update instances
                    if verbose:
                        print("Length of corresponding instances: ", len(set(corresponding_instances)))
                    self.log += f"Length of corresponding instances: {len(set(corresponding_instances))} \n"

                    if len(set(corresponding_instances)) > 1:
                        for instance in set(corresponding_instances):
                            # Don't split if ground
                            if instance == 0:
                                continue

                            # Don't split if intersection too small
                            intersection_mask = mask & other_tree_mask
                            if np.sum(intersection_mask) < 2:
                                continue

                            # Splitting
                            dict_monitoring["splitting_using_pca"] += 1
                            other_tree_mask = new_file.treeID == instance
                            intersection_mask = mask & other_tree_mask
                            if np.sum(intersection_mask) > 1:
                                mask, new_other_tree_mask = Pipeline.split_instances(new_file, mask, other_tree_mask)
                            else:
                                new_other_tree_mask = (other_tree_mask.astype(int) - intersection_mask.astype(int)).astype(bool)
                            # Check if splitted instances are still predicted as trees
                            tree_1 = pointCloud[mask]
                            tree_2 = pointCloud[new_other_tree_mask]
                            preds_classifier = fast_inference([tree_1, tree_2], args_classifier)

                            # Stop adding the new instance if any of the two are not predicted as tree anymore
                            if np.any(np.argmax(preds_classifier, axis=1) != 2):
                                dict_monitoring["not_a_tree_after_splitting"] += 1
                                is_new_tree = False
                                break

                    if is_new_tree:
                        dict_monitoring["new_instance"] += 1
                        # update classification
                        new_file.classification[mask] = 4

                        # update instance
                        new_file.treeID[mask] = id_new_tree

                        if verbose:
                            print("New tree with instance: ", id_new_tree, " and length: ", np.sum(mask))
                        self.log += f"New tree with instance: {id_new_tree}  and length: {np.sum(mask)} \n"
                        id_new_tree += 1
             
            # saving back original file and also to corresponding loop if flag set to
            new_file.write(original_file_src)
            if self.save_pseudo_labels_per_loop:
                os.makedirs(os.path.join(self.result_current_loop_dir, "pseudo_labels"), exist_ok=True)
                loop_file_src = os.path.join(self.result_current_loop_dir, "pseudo_labels", os.path.basename(original_file_src))
                new_file.write(loop_file_src)
            
        # removing unsused tiles from pseudo-label directory (so that it is not processed by the training phase)
        list_tiles_pseudo_labels = [x for x in os.listdir(self.result_pseudo_labels_dir) if x.endswith(self.file_format)]
        for tile_name in list_tiles_pseudo_labels:
            if tile_name not in self.tiles_to_process:
                os.remove(os.path.join(self.result_pseudo_labels_dir, tile_name))

        # Save monitoring
        print(dict_monitoring)
        pd.DataFrame(index=dict_monitoring.keys(), data=dict_monitoring.values(), columns=['count']).to_csv(os.path.join(self.result_current_loop_dir, 'pseudo_labels_monitoring.csv'), sep=';')
        
        # If flatten, create a flatten version of the pseudo_labels
        self.original_result_pseudo_labels_dir = ""
        if self.do_flatten:
            self.original_result_pseudo_labels_dir = self.result_pseudo_labels_dir
            self.result_pseudo_labels_dir = os.path.normpath(self.result_pseudo_labels_dir) + "_flatten"

            # if initialization, copy pseudo-labels to flatten
            if self.current_loop == 0:
                os.makedirs(self.result_pseudo_labels_dir, exist_ok=True)
                for tile in os.listdir(os.path.join(self.data_src, "flatten")):
                    destination_file = os.path.join(self.result_pseudo_labels_dir, tile.split('_flatten')[0] + '.laz')
                    shutil.copyfile(
                        os.path.join(self.data_src, 'flatten', tile),
                        destination_file
                    )
                shutil.copyfile(
                    os.path.join(self.original_result_pseudo_labels_dir, 'data_split_metadata.csv'),
                    os.path.join(self.result_pseudo_labels_dir, 'data_split_metadata.csv'),
                )

            for tile in [x for x in os.listdir(self.original_result_pseudo_labels_dir) if x.endswith('.laz')]:
                flatten_pseudo_labels_src = os.path.join(self.result_pseudo_labels_dir, tile)
                flatten_file = laspy.read(flatten_pseudo_labels_src)
                original_file = laspy.read(os.path.join(self.original_result_pseudo_labels_dir, tile))

                if self.current_loop == 0:
                    flatten_file.add_extra_dim(
                        laspy.ExtraBytesParams(
                            name='treeID',
                            type="f4",
                            description='Instance p-label'
                        )
                    )
                
                    # reset values of pseudo-labels
                    flatten_file.classification = np.zeros(len(flatten_file), dtype="f4")
                    flatten_file.treeID = np.zeros(len(flatten_file), dtype="f4")

                flatten_file.__setattr__('classification', original_file.classification)
                flatten_file.__setattr__('treeID', original_file.treeID)
                flatten_file.write(flatten_pseudo_labels_src)

    def process_row_single(self, coords_original_file_view, row_id):
        # Find matching points between original file and cluster
        mask = np.isin(coords_original_file_view, self.classified_clusters[row_id][0])

        return mask, self.classified_clusters[row_id][1]

    def process_row_multi(self, coords_original_file_view, row_id):
        # Find matching points between original file and cluster
        mask = np.isin(coords_original_file_view, self.multi_clusters[row_id][0])
        
        return mask, self.classified_clusters[row_id][1]

    def stats_on_tiles(self):
        """
        Compute classification results and panoptic quality metrics for each tile by processing segmentation outputs and comparing predicted and ground truth instances.

        Args:
            - None

        Returns:
            - None: Updates the inference metrics with classification results and panoptic quality metrics for each processed tile.
        """

        print("Computing stats on tiles")
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
            pred_src = ""
            if self.do_flatten:
                pred_src = os.path.join(self.data_src, 'preds_flatten', file.split('.')[0] + "_flatten_" + str(self.flatten_tile_size) +'m_out.' + self.file_format)
            else:
                pred_src = os.path.join(self.data_src, 'preds', file.split('.')[0] + '_out.' + self.file_format)
            tile_preds = laspy.read(pred_src)

            # match the original with the pred
            tile_original = Pipeline.remove_duplicates(tile_original)
            tile_preds = Pipeline.remove_duplicates(tile_preds)
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
        """
        Prepare the data for further processing by running a subprocess to convert sample data into the required format.

        Args:
            - verbose (bool, optional): Whether to display detailed progress and logging information. Defaults to True.

        Returns:
            - None: Executes a subprocess for data preparation.
        """

        print("Prepare data:")
        self.run_subprocess(
            src_script="/home/pdm/models/SegmentAnyTree/",
            script_name="./run_sample_data_conversion.sh",
            params= [self.result_pseudo_labels_dir],
            verbose=verbose
            )

    def train(self, verbose=True):
        """
        Train the model using the specified configuration and data, modifying configuration files and running the training subprocess.

        Args:
            - verbose (bool, optional): Whether to display detailed progress and logging information during training. Defaults to True.

        Returns:
            - None: Initiates model training and updates the model checkpoint path after completion.
        """

        # test if enough tiles available
        for type in ['train', 'test', 'val']:
            if len([x for x in os.listdir(os.path.join(self.result_pseudo_labels_dir, 'treeinsfused/raw/', os.path.split(os.path.abspath(self.result_pseudo_labels_dir))[-1])) if x.split('.')[0].endswith(type)]) == 0:
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
        )

        # run training script
        model_checkpoint = self.model_checkpoint_src if self.model_checkpoint_src != None else "/home/pdm/models/SegmentAnyTree/model_file"
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
        self.model_checkpoint_src = self.result_current_loop_dir

        if self.do_flatten:
            self.result_pseudo_labels_dir = self.original_result_pseudo_labels_dir

    def visualization(self, with_training=True, with_gt=False, with_groups=False):
        """
        Generate and save visualization plots for inference metrics, pseudo labels, and training results.

        Args:
            - with_training (bool, optional): Whether to include training-related visualizations. Defaults to True.
            - with_gt (bool, optional): Whether to include ground truth comparison visualizations. Defaults to False.

        Returns:
            - None: Saves visualizations to the specified directory.
        """

        print("Saving plots:")
        
        # creates location
        location_src = os.path.join(self.result_dir, "images")
        os.makedirs(location_src, exist_ok=True)

        # create plots
        show_inference_metrics(
            data_src=self.inference_metrics_src,
            src_location=os.path.join(location_src, 'inference_metrics.png'),
            show_figure=False,
            save_figure=True,
            )        
        show_inference_counts(
            data_src=self.inference_metrics_src,
            src_location=os.path.join(location_src, 'inference_count.png'),
            show_figure=False,
            save_figure=True,
            )
        show_pseudo_labels_evolution(
            data_folder=self.result_dir,
            src_location=os.path.join(location_src, 'pseudo_labels.png'),
            show_figure=False,
            save_figure=True,
        )
        if with_training:
            show_global_metrics(
                data_src=self.training_metrics_src,
                src_location=os.path.join(location_src, 'training_metrics.png'),
                show_figure=False,
                save_figure=True,
                )
            show_training_losses(
                data_src=self.training_metrics_src, 
                src_location=os.path.join(location_src, "training_losses.png"), 
                show_figure=False,
                save_figure=True, 
                )
            show_stages_losses(
                self.training_metrics_src, 
                src_location=os.path.join(location_src, "stages_losses.png"), 
                show_figure=False,
                save_figure=True, 
                )
        if with_gt:
            show_pseudo_labels_vs_gt(
                data_folder=self.result_dir,
                src_location=os.path.join(location_src, "peudo_labels_vs_gt.png"),
                show_figure=False,
                save_figure=True,
            )
        if with_groups:
            show_test_set(
                data_folder=self.result_dir,
                src_location=os.path.join(location_src, "peudo_labels_vs_gt.png"),
                cluster_csv_file=self.cfg.pipeline.inference.groups_csv_path,
                show_figure=False,
                save_figure=True,
            )

    def save_log(self, dest, clear_after=True, verbose=False):
        """
        Save the logs to the specified destination and optionally clear the log after saving.

        Args:
            - dest (str): The destination directory where the log should be saved.
            - clear_after (bool, optional): Whether to clear the log after saving. Defaults to True.
            - verbose (bool, optional): Whether to print the log saving progress. Defaults to False.

        Returns:
            - None: Saves the log to the specified destination and optionally clears it.
        """

        if verbose:
            print(f"Saving logs (of size {len(self.log)}) to : {dest}")
        
        os.makedirs(dest, exist_ok=True)

        with open(os.path.join(dest, "log.txt"), "w") as file:
            file.write(self.log)
        if clear_after:
            self.log = ""
