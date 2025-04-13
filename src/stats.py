import os
import numpy as np
import pandas as pd


def compute_classification_results(src_results):
    df_results = pd.read_csv(os.path.join(src_results, 'results.csv'), sep=';')

    return [x[0] for x in df_results.groupby('class').count().values]

    
if __name__ == '__main__':
    src = r"D:\PDM_repo\Github\PDM\data\dataset_tiles_100m\temp\loops\0\preds\color_grp_full_tile_100_out_split_instance\results"
    compute_classification_results(src)