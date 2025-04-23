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

def show_metric_over_epoch(df, metric_name, ax=None, save_figure=False, src_figure=None, show_figure=False):
    fig = None
    if ax == None:
        fig, ax = plt.figure()
    
    for stage in df.stage.unique():
        df_stage = df[df.stage == stage]
        ax.plot(df_stage.num_epoch, df_stage[metric_name], label=stage)

    if save_figure:
        if src_figure != None and fig != None:
            plt.savefig(fig, src_figure)
        else:
            raise UserWarning("When saving figure, the ax should not be precised and the src should be precise!")
        
    if show_figure and fig != None:
        plt.show()
        plt.close()

def show_metric_over_samples(df, metric_name, ax=None, save_figure=False, src_figure=None, show_figure=False):
    fig = None
    if ax == None:
        fig, ax = plt.figure()
    
    ax.plot(df.num_loop, df[metric_name])

    if save_figure:
        if src_figure != None and fig != None:
            plt.savefig(fig, src_figure)
        else:
            raise UserWarning("When saving figure, the ax should not be precised and the src should be precise!")
        
    if show_figure and fig != None:
        plt.show()
        plt.close()



def show_global_metrics(src_data, exclude_columns = ['num_loop', 'num_epoch', 'stage', 'map'], show_figure=True, save_figure=False):
    # load and prepare data
    df_data = pd.read_csv(src_data, sep=';')
    loops = df_data.num_loop.values
    epochs = df_data.num_epoch.values
    new_epochs = [100 + y * (max(epochs) - min(epochs) + 1) + x % 100 for x,y in zip(epochs,loops)]
    df_data.num_epoch = new_epochs

    # load metrics and set col and rows
    metrics = [metric for metric in df_data.columns if metric not in exclude_columns]
    n_metrics = len(metrics)
    n_cols = 4
    n_rows = (n_metrics + 1) // n_cols

    # plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        show_metric_over_epoch(df_data, metric, ax=axes[i])
        axes[i].set_title(metric)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # remove unused axes

    plt.tight_layout()
    plt.show()


def show_inference_counts(data_src):
    df_data = pd.read_csv(data_src, sep=';')
    print(df_data.columns)
    sums = df_data[["num_loop", "num_predictions", "num_garbage", "num_multi", "num_single"]].groupby('num_loop').sum()
    fractions = sums[["num_garbage", "num_multi", "num_single"]].div(sums["num_predictions"], axis=0)
    num_problematic = df_data[['num_loop', 'is_problematic']].groupby('num_loop').sum()
    num_empty = df_data[['num_loop', 'is_empty']].groupby('num_loop').sum()
    fig, axs = plt.subplots(2,2, figsize=(12,12))
    sns.lineplot(data=sums, ax=axs[0,0])
    sns.lineplot(data=fractions, ax=axs[0,1])
    sns.barplot(data=num_problematic, x="num_loop", y='is_problematic', ax=axs[1,0])
    sns.barplot(data=num_empty, x="num_loop", y='is_empty', ax=axs[1,1])
    # sns.lineplot(data=num_empty, ax=axs[1,1])
    axs[0,0].set_title('Count of the differente types of predictions')
    axs[0,1].set_title('Fraction over number of predictions')
    axs[1,0].set_title('Number of problematic samples')
    axs[1,1].set_title('Number of empty samples')
    plt.show()


def show_inference_metrics(data_src, metrics = ['PQ', 'SQ', 'RQ', 'Pre', 'Rec', 'mIoU']):
    abrev_to_name = {
        'PQ': "Panoptic Quality",
        'SQ': "Segmentation Quality",
        'RQ': "Recognition Quality",
        'Pre': "Precision",
        'Rec': "Recall",
        'mIoU': "Mean Intersection over Union",
    }
    df_data = pd.read_csv(data_src, sep=';')

    # plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        show_metric_over_samples(df_data, metric, ax=axes[i])
        axes[i].set_title(abrev_to_name[metric])
        if i % 2 == 0:
            axes[i].set_ylabel('Value [-]')
        if i in [4,5]:
            axes[i].set_xlabel('Loops [-]')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # remove unused axes

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    src_data_train = r"D:\PDM_repo\Github\PDM\results\trainings\20250420_181631_test\training_metrics.csv"
    src_data_inf = r"D:\PDM_repo\Github\PDM\results\trainings\20250420_181631_test\inference_metrics.csv"
    # print(loops)
    show_global_metrics(src_data_train)
    show_inference_counts(src_data_inf)
    show_inference_metrics(src_data_inf)