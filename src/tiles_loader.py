import os
import sys
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import laspy
from omegaconf import OmegaConf
import time
import threading
import json
import warnings
import zipfile

if __name__ == "__main__":
    sys.path.append(os.getcwd())

ENV = os.environ['CONDA_DEFAULT_ENV']
if ENV == "pdal_env":
    import pdal
    from src.splitting import split_instance
    from src.format_conversions import convert_all_in_folder 


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
    def remove_hanging_points(src_laz_in, src_laz_out, voxel_size=2, threshold=5):
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
        print(container)
        for _, point_id in tqdm(enumerate(range(points.shape[0])), total = points.shape[0]):
            full_pos = [0,0,0]
            for ax in range(3):
                for pos in range(len(voxel_indices[ax])):
                    if points[point_id, ax] > voxel_indices[ax][pos] and points[point_id, ax] < voxel_indices[ax][pos+1]:
                        full_pos[ax] = pos
                        break

            container[full_pos[0]][full_pos[1]][full_pos[2]].append(points[point_id])
            points_pos_in_container.append(full_pos)

        # find the isolated points
        isolated_points = []
        for _, point_id in tqdm(enumerate(range(points.shape[0])), total = points.shape[0]):
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
        src_sample = r"D:\PDM_repo\Github\PDM\data\full_dataset\selection\clusters_4\cluster_2\color_grp_full_tile_586.laz"
        new_file_src = os.path.basename(src_sample).split('.laz')[0] + 'voxel_size_2_isolated_th_5.laz'
        new_laz_src = os.path.join(os.path.dirname(src_sample), new_file_src)
        laz_in = laspy.read(new_laz_src)
        mask_isolated = laz_in.isolated == 0

        # remove points based on mask
        laz_in.points = laz_in.points[mask_isolated]
        laz_in.write(src_laz_out)

    # ===================================
    # === METHODS OF THE TILES LOADER ===
    # ===================================

    def tiling(self, verbose):
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

    def preprocess(self):
        # security
        self.list_tiles = [x for x in os.listdir(self.data_dest)]
        assert len(self.list_tiles) != 0

        # remove hanging points
        if self.tilesloader_conf.preprocess.do_remove_hanging_points:
            for tile in self.list_tiles:
                TilesLoader.remove_hanging_points(
                    src_laz_in=os.path.join(self.data_dest, tile),
                    src_laz_out=os.path.join(self.data_dest, tile),
                    voxel_size=2,
                    threshold=5,
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

    def evaluate(self, list_of_tiles_to_remove=[], verbose=True):
        # # prepare architecture
        # os.makedirs(os.path.join(run_src, ""), exist_ok=True)

        # load csv of clusters
        df_clusters = pd.read_csv(self.tilesloader_conf.evaluate.cluster_csv_path, sep=';')
        number_of_clusters = sorted(df_clusters.cluster_id.unique().tolist())
        print(df_clusters.head())

        # remove tiles if necessary
        if len(list_of_tiles_to_remove) > 0:
            df_clusters = df_clusters.loc[~df_clusters.tile_name.isin(list_of_tiles_to_remove)]
        lst_tiles = df_clusters.tile_name.values
        print(df_clusters.head())

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
        lst_tiles = lst_tiles[0:10]
        for _, loop in tqdm(enumerate(loops), total=len(loops), desc="Evaluating"):
            if verbose:
                print(f"Processing loop {loop+1}/{len(loops)}")
            # segment on tiles
            inference_res_src = os.path.join(self.tilesloader_conf.evaluate.run_src, str(loop), 'inference')
            os.makedirs(inference_res_src, exist_ok=True)
            for _, tile in tqdm(enumerate(lst_tiles), total=len(lst_tiles), desc="Infering on tiles", disable=verbose==False):
                # create architecture
                temp_folder = os.path.join(self.tilesloader_conf.evaluate.run_src, 'temp_inf')
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
                os.makedirs(temp_folder)
                shutil.copyfile(
                    os.path.join(self.tilesloader_conf.evaluate.cluster_src, tile),
                    os.path.join(temp_folder, tile),
                )
                # segment
                print(temp_folder)
                print(inference_res_src)
                return_code = self.run_subprocess(
                    src_script=self.segmenter_conf.root_model_src,
                    script_name="./run_oracle_pipeline.sh",
                    params= [os.path.join(self.root_src,temp_folder), os.path.join(self.root_src, inference_res_src)],
                    verbose=verbose
                )
                print(return_code)

            # quit()
            # classify on samples

            # add classification results in csv
            pass

        # plot evolution per cluster 
        pass



if __name__ == "__main__":
    time_start = time.time()
    cfg_tilesloader = OmegaConf.load("config/tiles_loader.yaml")
    cfg_segmenter = OmegaConf.load("config/segmenter.yaml")
    cfg_classifier = OmegaConf.load("config/classifier.yaml")
    cfg = OmegaConf.merge(cfg_tilesloader, cfg_segmenter, cfg_classifier)
    tiles_loader = TilesLoader(cfg)

    list_to_drop = ["color_grp_full_tile_568.laz", "color_grp_full_tile_504.laz"]

    tiles_loader.evaluate(list_to_drop, verbose=True)
    quit()

    if len(sys.argv) > 1:

        # tests and variable declaration
        assert len(sys.argv) == 3
        mode = sys.argv[1]
        verbose = True if sys.argv[2].lower() == 'true' else False
        assert mode in ["tiling", "trimming", "classification"]
        assert verbose in [True, False]
        
        # call function
        #print("Mode: ", mode, "\nverbose: ", verbose)
        #quit()

        if mode == "tiling":
            tiles_loader.tiling(verbose=verbose)
        elif mode == "trimming":
            tiles_loader.trimming(verbose=verbose)
        elif mode == "classification":
            tiles_loader.classify(verbose=verbose)
        elif mode == "evaluate":
            TilesLoader.e
        else:
            pass
        quit()
    #tiles_loader.tiling()
    #tiles_loader.trimming(verbose=False)
    tiles_loader.classify(verbose=True)
    
    delta_time = time.time() - time_start
    print(f"Process done in {delta_time} seconds")
