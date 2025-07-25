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
    Run the full processing pipeline for data preparation, segmentation, classification, and training.

    Args:
        - cfg (omegaconf.DictConfig): all necessary parameters for pipeline configuration, and processing options.

    Returns:
        - None: The function executes the pipeline with logging and file outputs but returns no values.
    """
    
    # fixing seed
    random.seed(42)

    # load pipepline arguments
    ROOT_SRC = cfg.pipeline.root_src

    # load data
    NUM_LOOPS = cfg.pipeline.num_loops
    DATA_SRC = os.path.join(cfg.dataset.project_root_src, cfg.dataset.data_src)

    TRAIN_FRAC = cfg.pipeline.train_frac
    TEST_FRAC = cfg.pipeline.test_frac
    VAL_FRAC = cfg.pipeline.val_frac

    # processes
    DO_REMOVE_HANGING_POINTS = cfg.pipeline.processes.do_remove_hanging_points
    DO_FLATTEN = cfg.pipeline.processes.do_flatten
    FLATTEN_TILE_SIZE = cfg.pipeline.processes.flatten_tile_size
    DO_CONTINUE_FROM_EXISTING = cfg.pipeline.preload.do_continue_from_existing

    # assertions
    assert TRAIN_FRAC + TEST_FRAC + VAL_FRAC == 1.0

    # start timer
    time_start_process = time()

    # preprocess
    if DO_REMOVE_HANGING_POINTS:
        pass

    if DO_FLATTEN and not DO_CONTINUE_FROM_EXISTING:
        os.makedirs(os.path.join(DATA_SRC, "originals"), exist_ok=True)
        flattening(DATA_SRC, os.path.join(DATA_SRC, "originals"), FLATTEN_TILE_SIZE, verbose_full=False)

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
        pipeline.segment(verbose=False)
        pipeline.save_log(pipeline.result_current_loop_dir, clear_after=False)

        # create csv for files referencing
        num_train = int(len(pipeline.tiles_to_process) * (TRAIN_FRAC + VAL_FRAC))
        train_test_split = random.sample(range(len(pipeline.tiles_to_process)), num_train)
        
        lst_split_data = []
        for id_f, f in enumerate(pipeline.tiles_to_process):
            new_row = [ f, 'ARPETTE']
            if id_f in train_test_split:
                new_row.append('train')
            else:
                new_row.append('test')
            lst_split_data.append(new_row)

        df_split_data = pd.DataFrame(columns=['path', 'folder', 'split'], data=lst_split_data)
        df_split_data.to_csv(os.path.join(pipeline.result_pseudo_labels_dir, 'data_split_metadata.csv'), sep=',', index=False)
        
        # classify
        pipeline.classify(verbose=False)
        pipeline.save_log(pipeline.result_current_loop_dir, clear_after=False)

        # create pseudo-labels
        pipeline.create_pseudo_labels(verbose=False)
        
        # compute stats on tiles
        pipeline.stats_on_tiles()

        # train
        pipeline.prepare_data(verbose=False)
        pipeline.train(verbose=False)

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
    pipeline.visualization()

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
