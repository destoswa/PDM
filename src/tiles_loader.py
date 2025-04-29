import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import laspy
from omegaconf import OmegaConf
import time
import threading
import pdal
import json
import warnings


class TilesLoader():
    def __init__(self, cfg):
        self.tilesloader_conf = cfg.tiles_loader
        self.segmenter_conf = cfg.segmenter
        self.data_src = self.tilesloader_conf.original_file_path
        self.data_dest = self.tilesloader_conf.tiles_destination
        self.list_tiles = []
        self.list_pack_of_tiles = []
        self.problematic_tiles = []
        # self.tiling = cfg.tiling
        # self.trimming = cfg.trimming
        # self.preprocess = cfg.preprocess
        
        assert os.path.exists(self.data_src)


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
        script_str = script_name
        # script_str = ['bash', script_name]
        if params:
            # for x in params:
                # script_str.append(str(x))
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

    #   _static methods used by the preprocess

    # ===================================
    # === METHODS OF THE TILES LOADER ===
    # ===================================

    def tiling(self):
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
        print("Computing the estimated number of tiles...")
        original_file = laspy.read(self.data_src)
        x_min = original_file.x.min()
        x_max = original_file.x.max()
        y_min = original_file.y.min()
        y_max = original_file.y.max() 
        expected_tiles = ((x_max - x_min) * (y_max - y_min)) // self.tilesloader_conf.tiling.tile_size ** 2
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
        #  _verify that all the files of the destination have the same extension
        if len(set([x.split('.')[-1] for x in os.listdir(self.data_dest)])) != 1:
            warnings.warn('It seems like the resulting folder contains files with different extensions!')
        #   _load
        self.list_tiles = [x for x in os.listdir(self.data_dest)]

        print("Tiling complete.")

    def trimming(self, verbose=True):
        # security
        if len(self.list_tiles) == 0:
            print("No tiles are loaded in the system!")
            answer = None
            while answer not in ['y', 'yes', 'n', 'no', '']:
                answer = input("Do you want to try and load them (y/n)?")
                if answer.lower() in ['y', 'yes', '']:
                    if len(set([x.split('.')[-1] for x in os.listdir(self.data_dest)])) > 1:
                        warnings.warn('It seems like the resulting folder contains files with different extensions!')
                    self.list_tiles = [x for x in os.listdir(self.data_dest)]
                    print('yes')
                elif answer.lower() in ['n', 'no']:
                    print("Stoping the process..")
                    quit()
                else:
                    print("wrong input.")
            if len(self.list_tiles) == 0:
                print("No files in the destination folder")
                quit()

        # creates pack of samples to infer on
        if self.tilesloader_conf.tiling.pack_size > 1:
            self.list_pack_of_tiles = [self.list_tiles[x:min(y,len(self.list_tiles))] for x, y in zip(
                range(0, len(self.list_tiles) - self.tilesloader_conf.tiling.pack_size, self.tilesloader_conf.tiling.pack_size),
                range(self.tilesloader_conf.tiling.pack_size, len(self.list_tiles), self.tilesloader_conf.tiling.pack_size),
                )]
            if self.list_pack_of_tiles[-1][-1] != self.list_tiles[-1]:
                self.list_pack_of_tiles.append(self.list_tiles[(len(self.list_pack_of_tiles)*self.tilesloader_conf.tiling.pack_size)::])
        else:
            self.list_pack_of_tiles = [[x] for x in self.list_tiles]

        # # select checkpoint
        # if self.model_checkpoint_src == "None":
        #     TilesLoader.change_var_val_yaml(
        #         src_yaml=self.inference.config_eval_src,
        #         var="checkpoint_dir",
        #         val="/home/pdm/models/SegmentAnyTree/model_file",
        #     )
        # else:
        #     TilesLoader.change_var_val_yaml(
        #         src_yaml=self.inference.config_eval_src,
        #         var="checkpoint_dir",
        #         val=os.path.join(self.cfg.pipeline.root_src, self.model_checkpoint_src),
        #     )

        # create temp folder
        temp_seg_src = os.path.join(self.tilesloader_conf.trimming.results_dest, 'temp_seg')
        if os.path.exists(temp_seg_src):
            shutil.rmtree(temp_seg_src)
        os.makedirs(temp_seg_src)

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
                print(temp_file_src)
                shutil.copyfile(original_file_src, temp_file_src)
            # quit()

            # segment on it
            # segmentation_results_dir = os.path.join(self.tilesloader_conf.trimming.results_dest, "segmented")
            # return_code = self.run_subprocess(
            #     src_script=self.segmenter_conf.root_model_src,
            #     script_name="./run_inference.sh",
            #     params= [temp_seg_src, segmentation_results_dir, True],
            #     verbose=verbose
            #     )
            segmentation_results_dir = os.path.join(self.tilesloader_conf.trimming.results_dest, "segmented")
            return_code = self.run_subprocess(
                src_script=self.segmenter_conf.root_model_src,
                script_name="./run_oracle_pipeline.sh",
                params= [temp_seg_src, segmentation_results_dir],
                verbose=verbose
                )
            
            # catch errors
            if return_code != 0:
                if verbose:
                    print(f"Problem with tiles:")
                    for file in pack:
                        print("\t", file)
                        self.problematic_tiles.append(file)
            else:
                # unzip results
                if verbose:
                    print("Unzipping results...")
                self.unzip_laz_files(
                    zip_path=os.path.join(segmentation_results_dir, "results.zip"),
                    extract_to=segmentation_results_dir,
                    delete_zip=True
                    )
                
            # removing temp file
            for file in os.listdir(temp_seg_src):
                os.remove(os.path.join(temp_seg_src, file))

    def preprocess(self):
        pass



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
    cfg_tilesloader = OmegaConf.load("config/tiles_loader.yaml")
    cfg_segmenter = OmegaConf.load("config/segmenter.yaml")
    cfg = OmegaConf.merge(cfg_tilesloader, cfg_segmenter)
    tiles_loader = TilesLoader(cfg)
    # tiles_loader.tiling()
    tiles_loader.trimming(verbose=True)
