import os
import sys
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import laspy
from omegaconf import OmegaConf
import time
import threading
import json
import warnings
import zipfile


# from multiprocess import Pool  # instead of concurrent.futures
# import concurrent.futures

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial

from splitting import split_instance
from format_conversions import convert_all_in_folder 

if __name__ == "__main__":
    sys.path.append(os.getcwd())

ENV = os.environ['CONDA_DEFAULT_ENV']
if ENV == "pdal_env":
    import pdal

# @staticmethod
# def remove_hanging_points_compare(points_pos_in_container, container, threshold, point_id):
#     num_neighboors = 0
#     pos = points_pos_in_container[point_id]
#     for dx in range(-1, 2):
#         for dy in range(-1, 2):
#             for dz in range(-1, 2):
#                 x = np.max([pos[0] + dx, 0])
#                 y = np.max([pos[1] + dy, 0])
#                 z = np.max([pos[2] + dz, 0])
#                 num_neighboors += len(container[x][y][z])
#     if num_neighboors < threshold:
#         return point_id
#         # isolated_points.append(point_id)
#     return -1

# def get_isolated_points(points, points_pos_in_container, container, threshold):
#     args = range(points.shape[0])
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         fn = partial(remove_hanging_points_compare,
#                      points_pos_in_container=points_pos_in_container,
#                      container=container,
#                      threshold=threshold)
#         results = list(tqdm(executor.map(fn, args), total=len(args), desc="Finding isolated points"))
#     return results

class TilesLoader():
    def __init__(self, cfg):
        self.cfg = cfg
        self.tilesloader_conf = cfg.tiles_loader
        self.segmenter_conf = cfg.segmenter
        self.classifier_conf = cfg.classifier
        self.root_src = self.tilesloader_conf.root_src
        self.data_src = self.tilesloader_conf.original_file_path
        self.trimming_method = self.tilesloader_conf.trimming.method
        self.trimming_tree_list = self.tilesloader_conf.trimming.tree_list
        self.not_yet_trim = True
        self.pack_size = self.tilesloader_conf.tiling.pack_size
        self.results_dest = self.tilesloader_conf.results_destination
        self.segmentation_results_dir = os.path.join(self.root_src, self.results_dest, "segmented")
        self.classification_results_dir = os.path.join(self.root_src, self.results_dest, "classified")
        self.data_dest = self.tilesloader_conf.tiles_destination
        self.tiles_to_remove = self.tilesloader_conf.evaluate.tiles_to_remove
        self.list_tiles = []
        self.list_pack_of_tiles = []
        self.problematic_tiles = []
        # self.tiling = cfg.tiling
        # self.trimming = cfg.trimming
        # self.preprocess = cfg.preprocess
        
        # assert os.path.exists(self.data_src)


    # ======================
    # === STATIC METHODS ===
    # ======================

    #   _general static methods
    @staticmethod
    def run_subprocess(src_script, script_name, params=None, verbose=True):
        # go at the root of the segmenter
        old_current_dir = os.getcwd()
        os.chdir(src_script)
        # construct command and run subprocess
        #script_str = script_name
        script_str = ['bash', script_name]
        if params:
            for x in params:
                script_str.append(str(x))
            #script_str += ' ' + ' '.join([str(x) for x in params])
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

        # Ensure the process has fully exited
        process.wait()
        if verbose:
            print(f"[INFO] Process completed with exit code {process.returncode}")

        # go back to original working dir
        os.chdir(old_current_dir)

        # return exit code
        return process.returncode

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
    
    #   _static methods used by the tiling:
    @staticmethod
    def monitor_progress(output_dir, expected_tiles, file_ext=".laz", poll_interval=0.5, thread=None):
        pbar = tqdm(total=expected_tiles, desc="Tiling progress")
        seen = set()
        while True:
            files = [f for f in os.listdir(output_dir) if f.endswith(file_ext)]
            new = set(files)
            progress = len(new)
            if (progress >= expected_tiles) or (thread and not thread.is_alive()):
                pbar.n = expected_tiles
                pbar.refresh()
                break
            if progress > len(seen):
                pbar.n = progress
                pbar.refresh()
            seen = new
            time.sleep(poll_interval)
        pbar.close()

    @staticmethod
    def run_pdal_pipeline(pipeline_json):
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()

    #   _static methods used by the trimming
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
    def remove_hanging_points(src_laz_in, src_laz_out, voxel_size=2, threshold=5, verbose=True):
        # voxelize the tile
        laz_in = laspy.read(src_laz_in)
        points = np.array(laz_in.xyz)
        voxel_size = 2
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        min = np.min(points, axis=0)
        max = np.max(points, axis=0)

        voxel_indices = []
        voxel_indices.append(np.arange(min[0], max[0] + voxel_size, voxel_size))
        voxel_indices.append(np.arange(min[1], max[1] + voxel_size, voxel_size))
        voxel_indices.append(np.arange(min[2], max[2] + voxel_size, voxel_size))

        container = {x:{y:{z:[] for z in range(len(voxel_indices[2]))} for y in range(len(voxel_indices[1]))} for x in range(len(voxel_indices[0]))}
        points_pos_in_container = []
        for _, point_id in tqdm(enumerate(range(points.shape[0])), total = points.shape[0], desc="Distribute points in voxels", disable=verbose==False):
            full_pos = [0,0,0]
            for ax in range(3):
                for pos in range(len(voxel_indices[ax])):
                    if points[point_id, ax] > voxel_indices[ax][pos] and points[point_id, ax] < voxel_indices[ax][pos+1]:
                        full_pos[ax] = pos
                        break

            container[full_pos[0]][full_pos[1]][full_pos[2]].append(points[point_id])
            points_pos_in_container.append(full_pos)

        # find the isolated points
        # results = get_isolated_points(points, points_pos_in_container, container, threshold)

        # with Pool() as pool:
        #     partialFunc = partial(self.remove_hanging_points_compare, points_pos_in_container, container, threshold)
        #     results = list(tqdm(pool.imap(partialFunc, range(points.shape[0])),
        #                         total=points.shape[0], desc="test"))

        # with ThreadPoolExecutor() as executor:
        #     partialFunc = partial(self.remove_hanging_points_compare, points_pos_in_container, container, threshold)
        #     # partialFunc = partial(self.test)
        #     results = list(tqdm(executor.map(partialFunc, range(points.shape[0])), total=points.shape[0], smoothing=0.9, desc="Updating pseudo-label"))




        isolated_points = []
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     partial_points_pos_in_container = partial(remove_hanging_points_compare, points_pos_in_container, container, threshold)
        #     args = range(points.shape[0])
        #     results = list(tqdm(executor.map(partial_points_pos_in_container, args), total=points.shape[0], smoothing=.9, desc="creating caching files"))
        # isolated_points = [x for x in results if x != -1]
        # print(f"Number of failing files: {len(num_fails)}")
        for _, point_id in tqdm(enumerate(range(points.shape[0])), total = points.shape[0], desc="Find isolated points", disable=verbose==False):
            num_neighboors = 0
            pos = points_pos_in_container[point_id]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        x = np.max([pos[0] + dx, 0])
                        y = np.max([pos[1] + dy, 0])
                        z = np.max([pos[2] + dz, 0])
                        num_neighboors += len(container[x][y][z])
            if num_neighboors < threshold:
                isolated_points.append(point_id)

        # create mask
        if 'isolated' not in list(laz_in.point_format.dimension_names):
            laz_in.add_extra_dim(
                laspy.ExtraBytesParams(
                    name='isolated',
                    type="f4",
                    description='Isolated points',
                    ),
                )
        laz_in.isolated = np.zeros(len(laz_in), dtype="f4")
        for iso_id in isolated_points:
            laz_in.isolated[iso_id] = 1
        # src_sample = r"D:\PDM_repo\Github\PDM\data\full_dataset\selection\clusters_4\cluster_2\color_grp_full_tile_586.laz"
        # new_file_src = os.path.basename(src_sample).split('.laz')[0] + 'voxel_size_2_isolated_th_5.laz'
        # new_laz_src = os.path.join(os.path.dirname(src_sample), new_file_src)
        # laz_in = laspy.read(new_laz_src)
        mask_isolated = laz_in.isolated == 0

        # remove points based on mask
        laz_in.points = laz_in.points[mask_isolated]
        laz_in.write(src_laz_out)

    # @staticmethod
    # def test(point_id):
    #     return point_id
    
    # # @staticmethod
    # def remove_hanging_points_compare(self, points_pos_in_container, container, threshold, point_id):
        num_neighboors = 0
        pos = points_pos_in_container[point_id]
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    x = np.max([pos[0] + dx, 0])
                    y = np.max([pos[1] + dy, 0])
                    z = np.max([pos[2] + dz, 0])
                    num_neighboors += len(container[x][y][z])
        if num_neighboors < threshold:
            return point_id
            # isolated_points.append(point_id)
        return -1
    
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

    # ===================================
    # === METHODS OF THE TILES LOADER ===
    # ===================================

    def tilling(self, verbose):
        print("Start tilling...")
        if os.path.exists(self.data_dest):
            answer = None
            while answer not in ['y', 'yes', 'n', 'no', ""]:
                answer = input("The resulting folder already exists. Do you want empty all its content (y/n)?")
                if answer.lower() in ['y', 'yes', '']:
                    shutil.rmtree(self.data_dest)
                    os.makedirs(self.data_dest, exist_ok=True)
                elif answer.lower() in ['n', 'no']:
                    print("Stoping the process..")
                    quit()
                else:
                    print("wrong input.")
        else:
            os.makedirs(self.data_dest, exist_ok=True)

        output_pattern = os.path.join(
            self.data_dest, 
            os.path.basename(self.data_src).split('.')[0] + "_tile_#.laz",
            )
        
        # compute the estimate number of tiles
        if verbose:
            print("Computing the estimated number of tiles...")
        original_file = laspy.read(self.data_src)
        x_min = original_file.x.min()
        x_max = original_file.x.max()
        y_min = original_file.y.min()
        y_max = original_file.y.max() 
        expected_tiles = ((x_max - x_min) * (y_max - y_min)) // self.tilesloader_conf.tiling.tile_size ** 2
        if verbose:
            print('Done!')

        # create the pdal command
        pipeline_json = {
            "pipeline": [
                self.data_src,
                {
                    "type": "filters.splitter",
                    "length": self.tilesloader_conf.tiling.tile_size  # Tile size in the X/Y direction
                },
                {
                    "type": "writers.las",
                    "filename": output_pattern
                }
            ]
        }

        # Launch PDAL pipeline in a separate thread
        print("Starting tiling (might take a few minutes to load the original file before starting:)")
        pipeline_thread = threading.Thread(target=TilesLoader.run_pdal_pipeline, args=(pipeline_json,))
        pipeline_thread.start()

        # Monitor the output folder in the main thread
        TilesLoader.monitor_progress(self.data_dest, expected_tiles=int(expected_tiles * 0.7), thread=pipeline_thread)

        pipeline_thread.join()

        # Load tiles path
        #   _verify that all the files of the destination have the same extension
        if len(set([x.split('.')[-1] for x in os.listdir(self.data_dest)])) != 1:
            warnings.warn('It seems like the resulting folder contains files with different extensions!')
        #   _load
        self.list_tiles = [x for x in os.listdir(self.data_dest)]

        print("Tiling complete.")

    def trimming(self, verbose=True):
        print("Start trimming...")
        if self.trimming_method == "tree":
            if self.not_yet_trim == True:
                self.pack_size = self.trimming_tree_list[0]
                self.trimming_tree_list.pop(0)
                self.not_yet_trim = False
            elif self.trimming_tree_list != [] and self.problematic_tiles != []:
                self.pack_size = self.trimming_tree_list[0]
                self.trimming_tree_list.pop(0)
                self.list_tiles = self.problematic_tiles
                self.problematic_tiles = []
            else:
                return

        # security
        self.list_tiles = [x for x in os.listdir(self.data_dest)]
        assert len(self.list_tiles) != 0

        # creates pack of samples to infer on
        if self.pack_size > 1:
            self.list_pack_of_tiles = [self.list_tiles[x:min(y,len(self.list_tiles))] for x, y in zip(
                range(0, len(self.list_tiles) - self.pack_size, self.pack_size),
                range(self.pack_size, len(self.list_tiles), self.pack_size),
                )]
            if self.list_pack_of_tiles[-1][-1] != self.list_tiles[-1]:
                self.list_pack_of_tiles.append(self.list_tiles[(len(self.list_pack_of_tiles)*self.pack_size)::])
        else:
            self.list_pack_of_tiles = [[x] for x in self.list_tiles]

        # select checkpoint
        TilesLoader.change_var_val_yaml(
                src_yaml=self.segmenter_conf.inference.config_eval_src,
                var="checkpoint_dir",
                val="/home/pdm/models/SegmentAnyTree/model_file",
            )

        # create temp folder
        temp_seg_src = os.path.join(self.root_src, self.results_dest, 'temp_seg')
        if os.path.exists(temp_seg_src):
            shutil.rmtree(temp_seg_src)
        os.makedirs(temp_seg_src)

        # pack_passed = [0,1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,1,0,1,0,1,0]
        # loops on samples:
        for _, pack in tqdm(enumerate(self.list_pack_of_tiles), total=len(self.list_pack_of_tiles), desc="Processing"):
            if verbose:
                print("===\tProcessing files: ")
                for file in pack:
                    print("\t", file)
                print("===")
            # copy files to temp folder
            for file in pack:
                original_file_src = os.path.join(self.data_dest, file)
                temp_file_src = os.path.join(os.path.join(temp_seg_src, file))
                # print(temp_file_src)
                shutil.copyfile(original_file_src, temp_file_src)
            # quit()

            # segment on it
            segmentation_results_dir = os.path.join(self.results_dest, "segmented")
            os.makedirs(self.segmentation_results_dir, exist_ok=True)
            # return_code = self.run_subprocess(
            #     src_script=self.segmenter_conf.root_model_src,
            #     script_name="./run_inference.sh",
            #     params= [temp_seg_src, self.segmentation_results_dir, True],
            #     verbose=verbose
            #     )
            return_code = self.run_subprocess(
                src_script=self.segmenter_conf.root_model_src,
                script_name="./run_oracle_pipeline.sh",
                params= [temp_seg_src, self.segmentation_results_dir],
                verbose=verbose
                )
            # quit()
            # test
            # return_code = pack_passed[id_pack]

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
                    zip_path=os.path.join(self.segmentation_results_dir, "results.zip"),
                    extract_to=self.segmentation_results_dir,
                    delete_zip=True
                    )
                # for file in pack:
                #     shutil.copyfile(
                #         os.path.join(self.data_dest, file),
                #         os.path.join(segmentation_results_dir, file)
                #     )
                
            # removing temp file
            for file in os.listdir(temp_seg_src):
                os.remove(os.path.join(temp_seg_src, file))

        # removing temp folder
        shutil.rmtree(temp_seg_src)

        # saving list of problematic
        with open(os.path.join(self.results_dest, 'problematic_tiles.txt'), 'w') as outfile:
            for item in self.problematic_tiles:
                outfile.write(f"{item}\n")

        # printing stuff for test
        print("State of the tree_list: ", self.trimming_tree_list)
        print("Files in the list: ", len(self.list_tiles))
        print("Failed tiles: ", len(self.problematic_tiles))
        print("Files in folder: ", len([x for x in os.listdir(os.path.join(segmentation_results_dir))]))

        if self.trimming_method == "tree":
            self.trimming(verbose=verbose)

    def preprocess(self, verbose=True):
        print("Start preprocessing...")
        # security
        self.list_tiles = [x for x in os.listdir(self.data_dest)]
        assert len(self.list_tiles) != 0

        # remove hanging points
        if self.tilesloader_conf.preprocess.do_remove_hanging_points:
            print("Removing hanging points...")
            for _, tile in tqdm(enumerate(self.list_tiles), total=len(self.list_tiles), desc="Processing"):
                TilesLoader.remove_hanging_points(
                    src_laz_in=os.path.join(self.data_dest, tile),
                    src_laz_out=os.path.join(self.data_dest, tile),
                    voxel_size=2,
                    threshold=5,
                    verbose=verbose
                )

    def classify(self, verbose=True):
        # create folder for classification results
        os.makedirs(self.classification_results_dir, exist_ok=True)

        # prepare results dict
        results_dist = {
            'tile_name': [],
            'garbage': [],
            'multi': [],
            'single': [],
        }
        list_files = [x for x in os.listdir(self.segmentation_results_dir) if x.endswith('.laz')]
        for _, file in tqdm(enumerate(list_files), total=len(list_files), desc="Classifying"):
            # load tile

            # create corresponding folder

            #
            tile_full_path = os.path.join(self.segmentation_results_dir, file)
            split_instance(tile_full_path, path_out=self.classification_results_dir, verbose=verbose)

            # convert instances to pcd
            dir_target = os.path.join(self.classification_results_dir, os.path.basename(tile_full_path).split('.')[0] + "_split_instance")
            convert_all_in_folder(
                src_folder_in=dir_target, 
                src_folder_out=os.path.join(dir_target, 'data'), 
                in_type="laz", 
                out_type='pcd',
                verbose=verbose
                )
            
            # makes predictions
            input_folder = dir_target
            output_folder = os.path.join(dir_target, 'data')
            code_return = self.run_subprocess(
                src_script=self.classifier_conf.root_model_src,
                script_name="./run_inference.sh",
                params= [input_folder, output_folder],
                verbose=verbose
                )
            if code_return != 0:
                print(f"WARNING! Subprocess for classification return code {code_return}!!")

            # remove data
            shutil.rmtree(os.path.join(dir_target, 'data'))

            # store distribution results
            df_file_results = pd.read_csv(os.path.join(dir_target, 'results/results.csv'), sep=';')
            counts = df_file_results.groupby('class').count()
            results_dist['tile_name'].append(''.join(file.split('_out')))
            for cat_num, cat_str in zip([0,1,2], ['garbage', 'multi', 'single']):
                if cat_num in counts.index:
                    results_dist[cat_str].append(counts.loc[cat_num].values[0])
                else: 
                    results_dist[cat_str].append(0)

        # save count results
        pd.DataFrame(results_dist).to_csv(
            os.path.join(self.results_dest, 'distribution_per_tile.csv'),
            sep=';',
            index=False,
        )

    def evaluate(self, verbose=True):

        # # load csv of clusters
        # df_clusters = pd.read_csv(self.tilesloader_conf.evaluate.cluster_csv_path, sep=';')
        # number_of_clusters = sorted(df_clusters.cluster_id.unique().tolist())

        # # remove tiles if necessary
        # if len(list_of_tiles_to_remove) > 0:
        #     df_clusters = df_clusters.loc[~df_clusters.tile_name.isin(list_of_tiles_to_remove)]
        # lst_tiles = df_clusters.tile_name.values
        # lst_tiles = []

        # num_per_cluster = 5
        # for cluster in number_of_clusters:
        #     list_clusters = df_clusters.loc[df_clusters.cluster_id == cluster].sample(n=num_per_cluster, random_state=42).tile_name.values

        #     if len(list_clusters) != num_per_cluster:
        #         raise ValueError(f"Not enough samples left in cluster {cluster}!")
        #     lst_tiles.append(list_clusters)
        #     # print(df_clusters.loc[df_clusters.cluster_id == cluster].sample(n=num_per_cluster))
        # lst_tiles_flatten = [x for row in lst_tiles for x in row]

        # # loop on loops:
        # loops = []
        # x = 0
        # while True:
        #     if str(x) in os.listdir(self.tilesloader_conf.evaluate.run_src):
        #         loops.append(x)
        #         x += 1
        #     else:
        #         break
        # if len(loops) == 0:
        #     print("No loops in run folder..")
        #     quit()

        # # process evolution per cluster
        # results_tot = {y: {x:{'garbage': [], 'multi': [], 'single': []} for x in loops} for y in range(len(lst_tiles))}

        # for id_group, group in enumerate(["Crouded flat", "Crouded steep", "Empty steep", "Empty flat"]):
        #     print("Group : ", id_group)
        #     for loop in loops:
        #         src_evaluation = os.path.join(self.root_src, self.tilesloader_conf.evaluate.run_src, str(loop), 'evaluation')
        #         list_folders = [x for x in os.listdir(src_evaluation) if os.path.isdir(os.path.join(src_evaluation, x)) and x.split('_out_split_instance')[0]+'.laz' in lst_tiles[id_group]]
        #         print(f"List folders in {src_evaluation}: {list_folders}")
        #         # load results per 
        #         for folder in list_folders:
        #             results_loop = pd.read_csv(os.path.join(src_evaluation, folder, "results/results.csv"), sep=';')
        #             print(results_loop.head())

        #             for cat_num, cat_name in enumerate(['garbage', 'multi', 'single']):
        #                 results_tot[id_group][loop][cat_name].append(len(results_loop.loc[results_loop['class'] == cat_num]))
        
        # results_agg = {
        #     x: {
        #         'garbage': [np.nanmean(results_tot[x][loop]["garbage"]) for loop in loops],
        #         'multi': [np.nanmean(results_tot[x][loop]["multi"]) for loop in loops],
        #         'single': [np.nanmean(results_tot[x][loop]["single"]) for loop in loops],
        #         } for x in range(len(lst_tiles))}

        # fig, axs = plt.subplots(2,2,figsize=(12,12))
        # axs = axs.flatten()
        # lst_titles = ['Crouded Flat', 'Crouded steep', 'Empty steep', 'Empty flat']
        # for id_ax, ax in enumerate(axs):
        #     df_results_agg = pd.DataFrame(results_agg[id_ax], index=range(len(loops)))
        #     ax.plot(df_results_agg)
        #     # ax.legend()
        #     ax.set_title(lst_titles[id_ax])

        # plt.savefig(os.path.join(self.root_src, self.tilesloader_conf.evaluate.run_src, 'test.png'))





        # quit()



        # # prepare architecture
        # os.makedirs(os.path.join(run_src, ""), exist_ok=True)

        # load csv of clusters
        df_clusters = pd.read_csv(self.tilesloader_conf.evaluate.cluster_csv_path, sep=';')
        number_of_clusters = sorted(df_clusters.cluster_id.unique().tolist())

        # remove tiles if necessary
        if len(self.tiles_to_remove) > 0:
            df_clusters = df_clusters.loc[~df_clusters.tile_name.isin(list_of_tiles_to_remove)]
        lst_tiles = df_clusters.tile_name.values
        lst_tiles = []

        num_per_cluster = 5
        for cluster in number_of_clusters:
            list_clusters = df_clusters.loc[df_clusters.cluster_id == cluster].sample(n=num_per_cluster, random_state=42).tile_name.values

            if len(list_clusters) != num_per_cluster:
                raise ValueError(f"Not enough samples left in cluster {cluster}!")
            lst_tiles.append(list_clusters)
            # print(df_clusters.loc[df_clusters.cluster_id == cluster].sample(n=num_per_cluster))
        lst_tiles_flatten = [x for row in lst_tiles for x in row]

        # loop on loops:
        loops = []
        x = 0
        while True:
            if str(x) in os.listdir(self.tilesloader_conf.evaluate.run_src):
                loops.append(x)
                x += 1
            else:
                break
        if len(loops) == 0:
            print("No loops in run folder..")
            quit()

        # lst_tiles = lst_tiles[0:10]
        # for _, loop in tqdm(enumerate(loops), total=len(loops), desc="Evaluating", disable=verbose==True):
        for loop in loops:
            # if verbose:
            print(f"Processing loop {loop+1}/{len(loops)}")
            loop_folder = os.path.join(self.tilesloader_conf.root_src, self.tilesloader_conf.evaluate.run_src, str(loop))
            # segment on tiles
            #   _change location of model
            TilesLoader.change_var_val_yaml(
                src_yaml=self.segmenter_conf.inference.config_eval_src,
                var="checkpoint_dir",
                val=loop_folder,
            )
            inference_res_src = os.path.join(self.root_src, self.tilesloader_conf.evaluate.run_src, str(loop), 'evaluation')
            os.makedirs(inference_res_src, exist_ok=True)

            #   _loop on tiles
            for _, tile in tqdm(enumerate(lst_tiles_flatten), total=len(lst_tiles_flatten), desc="Infering on tiles", disable=verbose==False):
                # create architecture
                temp_folder = os.path.join(self.root_src, self.tilesloader_conf.evaluate.run_src, 'temp_inf')
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
                os.makedirs(temp_folder)
                shutil.copyfile(
                    os.path.join(self.tilesloader_conf.evaluate.cluster_src, tile),
                    os.path.join(temp_folder, tile),
                )

                # segment
                return_code = self.run_subprocess(
                    src_script=self.segmenter_conf.root_model_src,
                    script_name="./run_oracle_pipeline.sh",
                    params= [temp_folder, temp_folder],
                    verbose=verbose
                )
                if return_code != 0:
                    if verbose:
                        print(f"Problem with tile {tile}:")
                    continue
                else:
                    # unzip results
                    if verbose:
                        print("Unzipping results...")
                    self.unzip_laz_files(
                        zip_path=os.path.join(temp_folder, "results.zip"),
                        extract_to=inference_res_src,
                        delete_zip=True
                        )
                
                # classify on samples
                tile_out = tile.split('.laz')[0] + '_out.laz'
                tile_out_path = os.path.join(inference_res_src, tile_out)
                split_instance(tile_out_path, verbose=verbose)

                # convert instances to pcd
                dir_target = tile_out_path.split('.laz')[0] + "_split_instance"
                convert_all_in_folder(
                    src_folder_in=dir_target, 
                    src_folder_out=os.path.join(dir_target, 'data'), 
                    in_type='laz', 
                    out_type='pcd',
                    verbose=verbose
                    )
                
                # makes predictions
                input_folder = dir_target
                output_folder = os.path.join(dir_target, 'data')
                code_return = self.run_subprocess(
                    src_script=self.classifier_conf.root_model_src,
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

        # process evolution per cluster
        results_tot = {y: {x:{'garbage': [], 'multi': [], 'single': []} for x in loops} for y in range(len(lst_tiles))}

        for id_group, group in enumerate(["Crouded flat", "Crouded steep", "Empty steep", "Empty flat"]):
            print("Group : ", id_group)
            for loop in loops:
                src_evaluation = os.path.join(self.root_src, self.tilesloader_conf.evaluate.run_src, str(loop), 'evaluation')
                list_folders = [x for x in os.listdir(src_evaluation) if os.path.isdir(os.path.join(src_evaluation, x)) and x.split('_out_split_instance')[0]+'.laz' in lst_tiles[id_group]]

                # load results per 
                for folder in list_folders:
                    results_loop = pd.read_csv(os.path.join(src_evaluation, folder, "results/results.csv"), sep=';')

                    for cat_num, cat_name in enumerate(['garbage', 'multi', 'single']):
                        results_tot[id_group][loop][cat_name].append(len(results_loop.loc[results_loop['class'] == cat_num]))
        
        results_agg = {
            x: {
                'garbage': [np.nanmean(results_tot[x][loop]["garbage"]) for loop in loops],
                'multi': [np.nanmean(results_tot[x][loop]["multi"]) for loop in loops],
                'single': [np.nanmean(results_tot[x][loop]["single"]) for loop in loops],
                } for x in range(len(lst_tiles))}

        fig, axs = plt.subplots(2,2,figsize=(12,12))
        axs = axs.flatten()
        lst_titles = ['Crouded Flat', 'Crouded steep', 'Empty steep', 'Empty flat']
        for id_ax, ax in enumerate(axs):
            df_results_agg = pd.DataFrame(results_agg[id_ax], index=range(len(loops)))
            ax.plot(df_results_agg)
            ax.legend()
            ax.set_title(lst_titles[id_ax])

        plt.savefig(os.path.join(self.root_src, self.tilesloader_conf.evaluate.run_src, 'test.png'))


        # # process evolution per cluster
        # results_tot = {x:{'garbage': [], 'multi': [], 'single': []} for x in loops}
        # for loop in loops:
        #     src_evaluation = os.path.join(self.root_src, self.tilesloader_conf.evaluate.run_src, str(loop), 'evaluation')
        #     # loop on instances:
        #     list_folders = [x for x in os.list_dir(src_evaluation) if os.path.isdir(x)]
        #     # load results per 
        #     for folder in list_folders:
        #         results_loop = pd.read_csv(os.path.join(src_evaluation, folder, "results.csv"))
        #         print(results_loop.groupby('class').count())
        #         distrib = results_loop.groupby('class').count().values
        #         results_tot[loop]['garbage'].append(distrib[0])
        #         results_tot[loop]['multi'].append(distrib[1])
        #         results_tot[loop]['single'].append(distrib[2])
        
        # results_agg = {loop: [
        #     np.mean(results_tot[loop]["garbage"]), 
        #     np.mean(results_tot[loop]["multi"]), 
        #     np.mean(results_tot[loop]["single"])
        #     ] for loop in loops}
        # df_results_agg = pd.DataFrame(results_agg, columns=['garbage', 'multi', 'single'])
        # # plot evolution per cluster
        # fig = plt.figure()
        # sns.lineplot(df_results_agg)
        # plt.show()

            


if __name__ == "__main__":
    # test = {
    #     0: [0.3, 0.4, 0.5],
    #     1: [0.3, 0.4, 0.5],
    #     2: [0.3, 0.4, 0.5],
    # }
    # df_test = pd.DataFrame(test)
    # fig = plt.figure()
    # plt.plot(df_test)
    # # sns.lineplot(df_test)
    # # plt.show()
    # quit()





    time_start = time.time()
    cfg_tilesloader = OmegaConf.load("config/tiles_loader.yaml")
    cfg_segmenter = OmegaConf.load("config/segmenter.yaml")
    cfg_classifier = OmegaConf.load("config/classifier.yaml")
    cfg = OmegaConf.merge(cfg_tilesloader, cfg_segmenter, cfg_classifier)
    tiles_loader = TilesLoader(cfg)

    # list_to_drop = ["color_grp_full_tile_568.laz", "color_grp_full_tile_504.laz"]
    # list_to_drop = [x for x in os.listdir(os.path.join(cfg_tilesloader.tiles_loader.root_src, cfg_tilesloader.tiles_loader.evaluate.run_src, "pseudo_labels")) if x.endswith('.laz')]

    # tiles_loader.evaluate(list_to_drop, verbose=True)
    # tiles_loader.preprocess()
    # quit()

    if len(sys.argv) > 1:

        # tests and variable declaration
        assert len(sys.argv) == 3
        mode = sys.argv[1]
        verbose = True if sys.argv[2].lower() == 'true' else False
        assert mode in ["tilling", "trimming", "classification"]
        assert verbose in [True, False]
        
        # call function
        #print("Mode: ", mode, "\nverbose: ", verbose)
        #quit()

        if mode == 'preprocess':
            tiles_loader.preprocess(verbose='verbose')
        if mode == "tilling":
            tiles_loader.tilling(verbose=verbose)
        elif mode == "trimming":
            tiles_loader.trimming(verbose=verbose)
        elif mode == "classification":
            tiles_loader.classify(verbose=verbose)
        elif mode == "evaluate":
            tiles_loader.evaluate(verbose=verbose)
        else:
            pass
        quit()
    #tiles_loader.tiling()
    #tiles_loader.trimming(verbose=False)
    # tiles_loader.classify(verbose=True)
    
    delta_time = time.time() - time_start
    print(f"Process done in {delta_time} seconds")
