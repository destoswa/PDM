import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import laspy
from tqdm import tqdm

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
    
    ax.plot(df.index, df[metric_name])

    if save_figure:
        if src_figure != None and fig != None:
            plt.savefig(fig, src_figure)
        else:
            raise UserWarning("When saving figure, the ax should not be precised and the src should be precise!")
        
    if show_figure and fig != None:
        plt.show()
        plt.close()



def show_global_metrics(src_data, exclude_columns = ['num_loop', 'num_epoch', 'stage', 'map'], src_location=None, show_figure=True, save_figure=False):
    # load and prepare data
    df_data = pd.read_csv(src_data, sep=';')

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
    if save_figure and src_location != None:
        plt.savefig(src_location)

    if show_figure:
        plt.show()


def show_inference_counts(data_src, src_location=None, show_figure=True, save_figure=False):
    df_data = pd.read_csv(data_src, sep=';')
    print("DATA:\n", df_data.head())
    sums = df_data[["num_loop", "num_predictions", "num_garbage", "num_multi", "num_single"]].groupby('num_loop').sum()
    fractions = sums[["num_garbage", "num_multi", "num_single"]].div(sums["num_predictions"], axis=0)
    num_problematic = df_data[['num_loop', 'is_problematic']].groupby('num_loop').sum()
    print(num_problematic)
    num_empty = df_data[['num_loop', 'is_empty']].groupby('num_loop').sum()
    fig, axs = plt.subplots(2,2, figsize=(12,12))
    sns.lineplot(data=sums, ax=axs[0,0])
    sns.lineplot(data=fractions, ax=axs[0,1])
    # sns.barplot(data=num_problematic, x="num_loop", y='is_problematic', ax=axs[1,0])
    # sns.barplot(data=num_empty, x="num_loop", y='is_empty', ax=axs[1,1])
    # sns.lineplot(data=num_empty, ax=axs[1,1])
    axs[0,0].set_title('Count of the differente types of predictions')
    axs[0,1].set_title('Fraction over number of predictions')
    # axs[1,0].set_title('Number of problematic samples')
    # axs[1,1].set_title('Number of empty samples')

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)

    if show_figure:
        plt.show()


def show_inference_metrics(data_src, metrics = ['PQ', 'SQ', 'RQ', 'Pre', 'Rec', 'mIoU'], src_location=None, show_figure=True, save_figure=False):
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
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        # average over all the samples
        df_data_metric = df_data[['num_loop', metric]]
        df_data_metric = df_data_metric[df_data_metric[metric] != 0]
        df_data_metric = df_data_metric.groupby("num_loop").mean()

        show_metric_over_samples(df_data_metric, metric, ax=axes[i])
        
        axes[i].set_title(abrev_to_name[metric])
        if i % 2 == 0:
            axes[i].set_ylabel('Value [-]')
        if i in [4,5]:
            axes[i].set_xlabel('Loops [-]')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # remove unused axes


    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)

    if show_figure:
        plt.show()


def show_pseudo_labels_evolution(data_folder, src_location=None, show_figure=True, save_figure=False):
    # load and generate data to show
    num_loop = 0
    count= {}
    change_from_previous = {}
    total_not_change = {}
    not_change_in_tile = {}
    previous_tiles = {}

    # finding the number of loops
    lst_loops = []
    while True:
        if not str(num_loop) in os.listdir(data_folder):
            break
        lst_loops.append(num_loop)
        num_loop += 1
    if num_loop == 0:
        print("No loop folder from which to extract the pseudo-labels")
        return   
    
    # processing each loop
    for _, num_loop in tqdm(enumerate(lst_loops), total=len(lst_loops), desc="Processing pseudo-labels for visualization"):
        count[num_loop] = {}
        change_from_previous[num_loop] = {}
        total_not_change[num_loop] = {}
        not_change_in_tile[num_loop] = {}
        for tile_src in os.listdir(os.path.join(data_folder, str(num_loop), "pseudo_labels")):
            tile = laspy.read(os.path.join(data_folder, str(num_loop), "pseudo_labels", tile_src))
            count[num_loop][tile_src] = [ np.sum(tile.classification == x) for x in [0, 1, 4]]
            if num_loop == 0:
                previous_tiles[tile_src] = tile.classification
                change_from_previous[num_loop][tile_src] = [0, 0, 0]
                total_not_change[num_loop][tile_src] = count[num_loop][tile_src]
                not_change_in_tile[num_loop][tile_src] = [True] * len(tile)
            else:
                total_not_change[num_loop][tile_src] = []
                change_from_previous[num_loop][tile_src] = []

                changes = tile.classification != previous_tiles[tile_src]
                not_change_in_tile[num_loop][tile_src] = list(np.array(changes) & np.array(not_change_in_tile[num_loop - 1][tile_src]))

                # loop on categories
                for cat in [0, 1, 4]:
                    mask = tile.classification == cat

                    # change from previous
                    change_from_previous[num_loop][tile_src].append(np.sum(changes[mask]))

                    # total no-change
                    total_not_change[num_loop][tile_src].append(np.sum(np.array(not_change_in_tile[num_loop][tile_src]) & np.array(mask)))
    
                previous_tiles[tile_src] = tile.classification 
     
    # aggregation
    categories = ['grey', 'ground', 'tree']
    count_agg= {x: [] for x in categories}
    change_from_previous_agg= {x: [] for x in categories}
    total_not_change_agg= {x: [] for x in categories}

    for num_loop in count.keys():
        for id_cat, cat in enumerate(categories):
            count_agg[cat].append(np.sum(np.array([tile_val for tile_val in count[num_loop].values()]), axis=0)[id_cat])
            change_from_previous_agg[cat].append(np.sum(np.array([tile_val for tile_val in change_from_previous[num_loop].values()]), axis=0)[id_cat])
            total_not_change_agg[cat].append(np.sum(np.array([tile_val for tile_val in total_not_change[num_loop].values()]), axis=0)[id_cat])
    
    # plotting semantic
    fig, axs = plt.subplots(2,2, figsize=(15,15))
    axs = axs.flatten()
    sns.lineplot(pd.DataFrame(count_agg), ax=axs[0])
    sns.lineplot(pd.DataFrame(change_from_previous_agg), ax=axs[1])
    sns.lineplot(pd.DataFrame(total_not_change_agg), ax=axs[2])
    fig.delaxes(axs[3])
    
    #   _titles and labels
    axs[0].set_title('count')
    axs[1].set_title('Change from previous loop')
    axs[2].set_title('Unchanges from beggining')
    
    plt.show()

    # plotting instances
    #...
    

if __name__ == '__main__':

    
    # Example data
    labels = ['Item 1', 'Item 2', 'Item 3']
    above = [0.7, 0.4, 0.6]   # e.g., % of category A
    below = [0.3, 0.6, 0.4]   # e.g., % of category B

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the bars
    ax.bar(x, above, label='Category A', color='steelblue')
    ax.bar(x, [-b for b in below], label='Category B', color='salmon')

    # Customizing the plot
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Percentage')
    ax.set_title('Diverging Bar Plot')
    ax.legend()

    plt.tight_layout()
    plt.show()
    quit()
    src_data_train = r"D:\PDM_repo\Github\PDM\results\trainings_saved\20250427_140314_test\training_metrics.csv"
    src_data_inf = r"D:\PDM_repo\Github\PDM\results\trainings_saved\20250427_140314_test\inference_metrics.csv"
    src_data_semantic = r"D:\PDM_repo\Github\PDM\results\trainings\20250507_084214_test"
    show_pseudo_labels_evolution(src_data_semantic)
    quit()
    # print(loops)
    # show_global_metrics(src_data_train)
    # show_inference_counts(src_data_inf)
    show_inference_metrics(src_data_inf)