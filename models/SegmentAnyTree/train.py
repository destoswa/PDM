import pandas as pd
import numpy as np
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer
import logging
import warnings
warnings.filterwarnings("ignore")


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # print(cfg.items())
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(OmegaConf.to_yaml(cfg))
    setattr(cfg, 'is_training',True)

    trainer = Trainer(cfg)

    metrics = trainer.train()

    # update metrics file
    df_metrics = pd.read_csv(cfg.train_metrics_src, sep=';')
    new_lines = {}
    for col in df_metrics.columns:
        new_lines[col] = []
    columns = df_metrics.columns
    for epoch, val in metrics.items():
        for stage, metric_vals in val.items():
            new_lines['num_loop'].append(cfg.current_loop)
            new_lines['num_epoch'].append(epoch)
            new_lines['stage'].append(stage)
            for el in [x for x in columns if x not in ['num_loop', 'num_epoch', 'stage']]:
                if el.lower() in [metric.lower() for metric in metric_vals['current_metrics'].keys()]:
                    # new_line.append(metric_vals['current_metrics'])
                    new_lines[el].append([metric_vals['current_metrics'][x] for x in metric_vals['current_metrics'].keys() if x.lower() == el.lower()][0])
                else:
                    new_lines[el].append(np.nan)
            # new_lines.append(new_line)
    # print("==========")
    # print("NEW LINES\n", new_lines)
    # print("==========")
    df_new_lines = pd.DataFrame(new_lines)
    # print("==========")
    # print("NEW DATAFRAME")
    # print(df_new_lines)
    # print("==========")
    df_metrics = pd.concat([df_metrics, df_new_lines], axis=0)
    df_metrics.to_csv(cfg.train_metrics_src, sep=';', index=False)

    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
