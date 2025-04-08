import os
import shutil
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import pdal

from src.format_conversions import convert_all_in_folder
from src.pipeline import Pipeline


def main(cfg):
    # load data
    data_src = cfg.dataset.data_src
    # print(OmegaConf.to_yaml(cfg))

    # create pipeline
    pipeline = Pipeline(cfg)    

    # segment
    # pipeline.segment()
    return

    # classify
    # pipeline.classify()

    # create pseudo-labels
    # pipeline.create_pseudo_labels()

    # train
    pipeline.train()

    # visualization


if __name__ == "__main__":
    cfg_dataset = OmegaConf.load('./config/dataset.yaml')
    cfg_preprocess = OmegaConf.load('./config/preprocessing.yaml')
    cfg_pipeline = OmegaConf.load('./config/pipeline.yaml')
    cfg_classifier = OmegaConf.load('./config/classifier.yaml')
    cfg_segmenter = OmegaConf.load('./config/segmenter.yaml')
    cfg = OmegaConf.merge(cfg_dataset, cfg_preprocess, cfg_pipeline, cfg_classifier, cfg_segmenter)
    main(cfg)
