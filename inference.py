import os
import shutil
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import random
from time import time
from src.pipeline import Pipeline
from src.preprocessing import flattening


def main(cfg):
    """
    Evaluate a given pipeline on a given ground truth set

    Args:
        - cfg (object): Configuration object containing pipeline parameters and paths.

    Returns:
        - None: Runs the entire pipeline, processing data through multiple stages and saving results, logs, and metrics.
    """

    # fixing seed
    random.seed(42)

    # load and change pipepline arguments
    ROOT_SRC = cfg.pipeline.root_src
    TARGET_RUN = cfg.pipeline.inference.src_existing
    cfg.pipeline.results_root_src = cfg.pipeline.inference.results_root_src
    cfg.pipeline.preload.do_continue_from_existing = False
    cfg.pipeline.result_src_name_suffixe = cfg.pipeline.inference.result_src_name_suffixe
    
    NUM_LOOPS = 0
    while str(NUM_LOOPS) in os.listdir(TARGET_RUN):
        NUM_LOOPS += 1
    if NUM_LOOPS == 0:
        raise ValueError("There is no existing loops in the project you are trying to start from!!")
    else:
        cfg.pipeline.num_loops = NUM_LOOPS

    # load data
    DATA_SRC = os.path.join(ROOT_SRC, cfg.pipeline.inference.src_data)
    cfg.dataset.data_src = DATA_SRC

    # processes
    DO_FLATTEN = cfg.pipeline.processes.do_flatten
    FLATTEN_TILE_SIZE = cfg.pipeline.processes.flatten_tile_size

    # start timer
    time_start_process = time()

    if DO_FLATTEN:
        os.makedirs(os.path.join(DATA_SRC, "originals"), exist_ok=True)
        flattening(DATA_SRC, os.path.join(DATA_SRC, "originals"), FLATTEN_TILE_SIZE)

    # create pipeline
    pipeline = Pipeline(cfg) 

    # start looping
    for loop in range(pipeline.current_loop, NUM_LOOPS):
        print(f"===== LOOP {loop + 1} / {NUM_LOOPS} =====")
        time_start_loop = time()
        pipeline.current_loop = loop
        pipeline.result_current_loop_dir = os.path.join(pipeline.result_dir, str(loop))

        # prepare architecture
        os.makedirs(os.path.join(DATA_SRC, f'loops/{loop}/'), exist_ok=True)
        
        list_tiles = [x for x in os.listdir(DATA_SRC) if x.endswith(pipeline.file_format)]
        for tile in list_tiles:
            if DO_FLATTEN:
                shutil.copyfile(os.path.join(DATA_SRC, 'originals', tile), os.path.join(DATA_SRC, f"loops/{loop}", tile))
            else:
                shutil.copyfile(os.path.join(DATA_SRC, tile), os.path.join(DATA_SRC, f"loops/{loop}", tile))

        if DO_FLATTEN:
            if os.path.exists(os.path.join(DATA_SRC, f'loops/{loop}/flatten')):
                shutil.rmtree(os.path.join(DATA_SRC, f'loops/{loop}/flatten'))
            shutil.copytree(os.path.join(DATA_SRC, 'flatten'), os.path.join(DATA_SRC, f'loops/{loop}/flatten'))

        pipeline.data_src = os.path.join(DATA_SRC, f'loops/{loop}/')
        pipeline.preds_src = os.path.join(pipeline.data_src, 'preds')
        
        # prepare states
        list_tiles_names = [x for x in os.listdir(os.path.join(pipeline.root_src, pipeline.data_src)) if x.endswith(pipeline.file_format)]
        loop_tiles_state = {
            "name": list_tiles_names,
            "num_loop": loop * np.ones((len(list_tiles_names))),
            "is_problematic": [int(x in pipeline.problematic_tiles) for x in list_tiles_names],
            "is_empty": np.zeros((len(list_tiles_names))),
            "num_predictions": np.zeros((len(list_tiles_names))),
            "num_garbage": np.zeros((len(list_tiles_names))),
            "num_multi": np.zeros((len(list_tiles_names))),
            "num_single": np.zeros((len(list_tiles_names))),
            "PQ": np.zeros((len(list_tiles_names))),
            "SQ": np.zeros((len(list_tiles_names))),
            "RQ": np.zeros((len(list_tiles_names))),
            "Pre": np.zeros((len(list_tiles_names))),
            "Rec": np.zeros((len(list_tiles_names))),
            "mIoU": np.zeros((len(list_tiles_names))),
        }
        pipeline.inference_metrics = pd.concat([pipeline.inference_metrics, pd.DataFrame(loop_tiles_state)], axis=0)

        # preprocess
        print("preprocessing...")
        pipeline.preprocess(verbose=False)

        # segment
        pipeline.model_checkpoint_src = os.path.join(ROOT_SRC, TARGET_RUN, str(loop))
        pipeline.segment(verbose=False)
        pipeline.save_log(pipeline.result_current_loop_dir, clear_after=False)

        # create dummy csv for files referencing
        df_split_data = pd.DataFrame()
        df_split_data.to_csv(os.path.join(pipeline.result_pseudo_labels_dir, 'data_split_metadata.csv'), sep=',', index=False)
        
        # classify
        pipeline.classify(verbose=False)
        pipeline.save_log(pipeline.result_current_loop_dir, clear_after=False)

        # create pseudo-labels
        pipeline.create_pseudo_labels(verbose=False)
        
        # compute stats on tiles
        pipeline.stats_on_tiles()

        if DO_FLATTEN:
            pipeline.result_pseudo_labels_dir = pipeline.original_result_pseudo_labels_dir

        # save logs
        pipeline.save_log(pipeline.result_current_loop_dir, clear_after=True)

        delta_time_loop = time() - time_start_loop
        hours = int(delta_time_loop // 3600)
        min = int((delta_time_loop - 3600 * hours) // 60)
        sec = int(delta_time_loop - 3600 * hours - 60 * min)
        print(f"==== Loop done in {hours}:{min}:{sec} ====")

        # save states info
        pipeline.inference_metrics.to_csv(
            os.path.join(
                ROOT_SRC, 
                pipeline.result_dir,
                "inference_metrics.csv"),
            sep=';',
            index=False,
        )

    # show metrics
    pipeline.visualization(
        with_training=False, 
        with_gt=cfg.pipeline.inference.is_gt, 
        with_groups=cfg.pipeline.inference.with_groups,
        )

    # show time to full process
    delta_time_process = time() - time_start_process
    hours = int(delta_time_process // 3600)
    min = int((delta_time_process - 3600 * hours) // 60)
    sec = int(delta_time_process - 3600 * hours - 60 * min)
    print(f"======\nFULL PROCESS DONE IN {hours}:{min}:{sec}\n======")


if __name__ == "__main__":
    cfg_dataset = OmegaConf.load('./config/dataset.yaml')
    cfg_pipeline = OmegaConf.load('./config/pipeline.yaml')
    cfg_classifier = OmegaConf.load('./config/classifier.yaml')
    cfg_segmenter = OmegaConf.load('./config/segmenter.yaml')
    cfg = OmegaConf.merge(cfg_dataset, cfg_pipeline, cfg_classifier, cfg_segmenter)

    main(cfg)