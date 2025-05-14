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

    #   _static methods used by the preprocess

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

            # print("No tiles are loaded in the system!")
            # answer = None
            # while answer not in ['y', 'yes', 'n', 'no', '']:
            #     answer = input("Do you want to try and load them (y/n)?")
            #     if answer.lower() in ['y', 'yes', '']:
            #         if len(set([x.split('.')[-1] for x in os.listdir(self.data_dest)])) > 1:
            #             warnings.warn('It seems like the resulting folder contains files with different extensions!')
            #         self.list_tiles = [x for x in os.listdir(self.data_dest)]
            #         print('yes')
            #     elif answer.lower() in ['n', 'no']:
            #         print("Stoping the process..")
            #         quit()
            #     else:
            #         print("wrong input.")
            # if len(self.list_tiles) == 0:
            #     print("No files in the destination folder")
            #     quit()

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
            quit()
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
        pass

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
            results_dist['tile_name'].append(file)
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
            # for file in os.listdir(output_folder):
            #     if file.endswith('.pcd'):
            #         os.remove(os.path.join(output_folder, file))

    # ===============================================
    # ======== for the future if enough time ========
    # ===============================================

    def make_stats(self):
        pass

    def filter(self):
        pass

    def split(self):
        pass

if __name__ == "__main__":    
    time_start = time.time()
    cfg_tilesloader = OmegaConf.load("config/tiles_loader.yaml")
    cfg_segmenter = OmegaConf.load("config/segmenter.yaml")
    cfg_classifier = OmegaConf.load("config/classifier.yaml")
    cfg = OmegaConf.merge(cfg_tilesloader, cfg_segmenter, cfg_classifier)
    tiles_loader = TilesLoader(cfg)

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
        else:
            pass
        quit()
    #tiles_loader.tiling()
    #tiles_loader.trimming(verbose=False)
    tiles_loader.classify(verbose=True)
    
    delta_time = time.time() - time_start
    print(f"Process done in {delta_time} seconds")
