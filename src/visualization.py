import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Ideas of plots:
    - for each global metric, show the evolution along the epochs (hoping that a drop in losses should happen between the loops)
    - for each tile's metric: 
        - show the per/tile metric evolution
        - show the average metric evolution along the loops
"""


def show_global_metrics(src_data, stage="train", show=True, save=False):
    data = pd.read_csv(src_data, sep=';')
    print(data.head())



if __name__ == '__main__':
    src_data = ""