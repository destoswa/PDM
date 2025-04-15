import pandas as pd
import numpy as np
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer
import logging


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    setattr(cfg, 'is_training',True)
    trainer = Trainer(cfg)
    # return
    metrics = trainer.train()
    print("=====\nMETRICS:\n", metrics)
    # update metrics file
    # df_metrics = pd.read_csv(cfg.train_metrics_src, sep=';')
    # new_lines = []
    # for epoch, val
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
