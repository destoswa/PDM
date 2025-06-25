import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import laspy
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

sys.path.append("D:/PDM_repo/Github/PDM")
from src.metrics import compute_classification_results, compute_panoptic_quality, compute_mean_iou

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
        ax.plot(np.array(df_stage.num_epoch), np.array(df_stage[metric_name]), label=stage)

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
    
    ax.plot(np.array(df.index), np.array(df[metric_name]))

    if save_figure:
        if src_figure != None and fig != None:
            plt.savefig(fig, src_figure)
        else:
            raise UserWarning("When saving figure, the ax should not be precised and the src should be precise!")
        
    if show_figure and fig != None:
        plt.show()
        plt.close()



def show_global_metrics(data_src, exclude_columns = ['num_loop', 'num_epoch', 'stage', 'map'], src_location=None, show_figure=True, save_figure=False):
    # load and prepare data
    df_data = pd.read_csv(data_src, sep=';')

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


def show_training_losses(data_src, src_location=None, show_figure=True, save_figure=False):
    # load and prepare data
    df_data = pd.read_csv(data_src, sep=';')
    df_data = df_data.loc[df_data.stage == 'train']
    sublosses_names = ['offset_norm_loss', 'offset_dir_loss', 'ins_loss', 'ins_var_loss', 'ins_dist_loss', 'ins_reg_loss', 'semantic_loss', 'score_loss']

    # plot
    fig = plt.figure(figsize=(10,6))
    plt.plot(np.array(df_data.index), np.array(df_data.loss), label='loss', linewidth=2.5)
    for loss_name in sublosses_names:
        plt.plot(np.array(df_data.index), np.array(df_data[loss_name]), label=loss_name, linewidth=1.3, alpha=0.3)
    plt.xlabel("Epoch [-]")
    plt.ylabel("Loss value [-]")
    plt.title("Evolution of losses by SegmentAnyTree")
    plt.legend(ncol=3)

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)

    if show_figure:
        plt.show()


def show_stages_losses(data_src, exclude_columns = ['num_loop', 'num_epoch', 'stage', 'map'], src_location=None, show_figure=True, save_figure=False):
    # load and prepare data
    df_data = pd.read_csv(data_src, sep=';')
    stages = df_data.stage.unique()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ['Training loss', 'Validation loss', 'Testing loss']
    # plot
    fig = plt.figure(figsize=(10,6))
    for id_stage, stage in enumerate(stages):
        df_subdata = df_data.loc[df_data.stage == stage]
        plt.plot(np.array(df_subdata.index), np.array(df_subdata.loss), linewidth=1.5, alpha=0.3, color=colors[id_stage])

        # smoothing
        n = 5
        num_rep = 5
        y_smooth = uniform_filter1d(np.array(df_subdata.loss), size=n)
        for i in range(num_rep - 1):
            y_smooth = uniform_filter1d(y_smooth, size=n)
        plt.plot(np.array(df_subdata.index), y_smooth, label=labels[id_stage], linewidth=2.5, color=colors[id_stage])
    plt.xlabel("Epoch [-]")
    plt.ylabel("Loss value [-]")
    plt.title("Evolution of losses per set")
    plt.legend()

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)

    if show_figure:
        plt.show()


def show_inference_counts(data_src, src_location=None, show_figure=True, save_figure=False):
    df_data = pd.read_csv(data_src, sep=';')
    averages = df_data[["num_loop", "num_predictions", "num_garbage", "num_multi", "num_single"]].groupby('num_loop').mean()
    fractions = averages[["num_garbage", "num_multi", "num_single"]].div(averages["num_predictions"], axis=0)
    # num_problematic = df_data[['num_loop', 'is_problematic']].groupby('num_loop').sum()
    # num_empty = df_data[['num_loop', 'is_empty']].groupby('num_loop').sum()

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs = axs.flatten()
    # for i, data in enumerate([sums.drop('num_predictions', axis=1), fractions, num_problematic, num_empty]):
    for i, data in enumerate([averages.drop('num_predictions', axis=1), fractions]):
        df = pd.DataFrame(data)
        for col in df.columns:
            axs[i].plot(np.array(df.index), np.array(df[col]), label=col)
            axs[i].legend()
    axs[0].set_title('Count of the differente types of predictions')
    axs[1].set_title('Fraction over number of predictions')
    # axs[2].set_title('Number of problematic samples')
    # axs[3].set_title('Number of empty samples')
    
    # set limits
    axs[0].set_ylim([0,np.max(averages.drop('num_predictions', axis=1).max().values)*1.1])
    axs[1].set_ylim([0,1])

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)

    if show_figure:
        plt.show()


def show_problematic_empty(data_src, src_location=None, show_figure=True, save_figure=False):
    df_data = pd.read_csv(data_src, sep=';')
    # sums = df_data[["num_loop", "num_predictions", "num_garbage", "num_multi", "num_single"]].groupby('num_loop').sum()
    # fractions = sums[["num_garbage", "num_multi", "num_single"]].div(sums["num_predictions"], axis=0)
    num_problematic = df_data[['num_loop', 'is_problematic']].groupby('num_loop').sum()
    num_empty = df_data[['num_loop', 'is_empty']].groupby('num_loop').sum()

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs = axs.flatten()
    # for i, data in enumerate([sums.drop('num_predictions', axis=1), fractions, num_problematic, num_empty]):
    for i, data in enumerate([num_problematic, num_empty]):
        df = pd.DataFrame(data)
        for col in df.columns:
            axs[i].plot(np.array(df.index), np.array(df[col]), label=col)
            axs[i].legend()
    # axs[0].set_title('Count of the differente types of predictions')
    # axs[1].set_title('Fraction over number of predictions')
    axs[0].set_title('Number of problematic samples')
    axs[1].set_title('Number of empty samples')
    
    # # set limits
    # print(np.max(sums.drop('num_predictions', axis=1).max().values))

    # axs[0].set_ylim([0,np.max(sums.drop('num_predictions', axis=1).max().values)*1.1])
    # axs[1].set_ylim([0,1])

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)

    if show_figure:
        plt.show()


def show_inference_metrics(data_src, metrics = ['PQ', 'SQ', 'RQ', 'Pre', 'Rec'], src_location=None, show_figure=True, save_figure=False):
    abrev_to_name = {
        'PQ': "Panoptic Quality",
        'SQ': "Segmentation Quality",
        'RQ': "Recognition Quality",
        'Pre': "Precision",
        'Rec': "Recall",
        # 'mIoU': "Mean Intersection over Union",
    }
    df_data = pd.read_csv(data_src, sep=';')
    df_data = df_data.loc[df_data.num_loop != 0]

    # plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharex=True, sharey=False)
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
    count_sem = {}
    count_inf = {}
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
        quit()   

    # processing each loop
    for _, num_loop in tqdm(enumerate(lst_loops), total=len(lst_loops), desc="Processing pseudo-labels for visualization"):
        count_sem[num_loop] = {}
        change_from_previous[num_loop] = {}
        total_not_change[num_loop] = {}
        not_change_in_tile[num_loop] = {}
        
        count_inf[num_loop] = []
        for tile_src in os.listdir(os.path.join(data_folder, str(num_loop), "pseudo_labels")):
            tile = laspy.read(os.path.join(data_folder, str(num_loop), "pseudo_labels", tile_src))
            count_sem[num_loop][tile_src] = [ np.sum(tile.classification == x) for x in [0, 1, 4]]
            count_inf[num_loop].append(len(set(tile.treeID)))
            if tile == "color_grp_full_tile_270.laz":
                print(len(set(tile.treeID)), ' - ', set(tile.treeID))
            if num_loop == 0:
                previous_tiles[tile_src] = tile.classification
                change_from_previous[num_loop][tile_src] = [0, 0, 0]
                total_not_change[num_loop][tile_src] = count_sem[num_loop][tile_src]
                not_change_in_tile[num_loop][tile_src] = [True] * len(tile)
            else:
                total_not_change[num_loop][tile_src] = []
                change_from_previous[num_loop][tile_src] = []

                changes = tile.classification != previous_tiles[tile_src]
                not_change_in_tile[num_loop][tile_src] = list(~np.array(changes) & np.array(not_change_in_tile[num_loop - 1][tile_src]))

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
    count_sem_agg = {x: [] for x in categories}
    count_inf_agg = []
    change_from_previous_agg = {x: [] for x in categories}
    total_not_change_agg = {x: [] for x in categories}

    for num_loop in count_sem.keys():
        count_inf_agg.append(np.mean(list(count_inf[num_loop])))
        for id_cat, cat in enumerate(categories):
            count_sem_agg[cat].append(np.mean([tile_val for tile_val in count_sem[num_loop].values()], axis=0)[id_cat])
            if num_loop > 0:
                change_from_previous_agg[cat].append(np.mean([tile_val for tile_val in change_from_previous[num_loop].values()], axis=0)[id_cat])
            total_not_change_agg[cat].append(np.mean([tile_val for tile_val in total_not_change[num_loop].values()], axis=0)[id_cat])
    
    # visualizing
    fig, axs = plt.subplots(2,2, figsize=(12,12))
    axs = axs.flatten()
    for i, data in enumerate([count_sem_agg, change_from_previous_agg, total_not_change_agg, count_inf_agg]):
        df = pd.DataFrame(data)
        for col in df.columns:
            axs[i].plot(np.array(df.index), np.array(df[col]), label=col)
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    #   _titles and labels
    axs[0].set_title('Count per semantic category')
    axs[1].set_title('Change from previous loop')
    axs[2].set_title('Unchanges from beggining')
    axs[3].set_title('Number of instances')
    
    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)

    if show_figure:
        plt.show()

    # plotting instances
    #...
    

def show_pseudo_labels_vs_gt(data_folder, src_location=None, metrics = ['PQ', 'SQ', 'RQ', 'Pre', 'Rec'], compute_metrics=False, show_figure=True, save_figure=False):

    abrev_to_name = {
        'PQ': "Panoptic Quality",
        'SQ': "Segmentation Quality",
        'RQ': "Recognition Quality",
        'Pre': "Precision",
        'Rec': "Recall",
    }

    df_metrics = None
    if compute_metrics:
        # finding the number of loops
        lst_loops = []
        num_loops = 0
        while True:
            if not str(num_loops) in os.listdir(data_folder):
                break
            lst_loops.append(num_loops)
            num_loops += 1
        if num_loops == 0:
            print("No loop folder from which to extract the pseudo-labels")
            quit()
        
        # loop on loops:
        metrics = {metric:[] for metric in ['loop', 'PQ', 'SQ', 'RQ', 'Rec', 'Pre']}
        for _, loop in tqdm(enumerate(range(num_loops)), total=num_loops, desc="Computing metrics on gt"):
            src_pseudo_labels = os.path.join(data_folder, str(loop), "pseudo_labels")
            for tile_src in os.listdir(src_pseudo_labels):
                tile = laspy.read(os.path.join(src_pseudo_labels, tile_src))
                gt_instances = tile.gt_instance
                pred_instances = tile.treeID
                PQ, SQ, RQ, tp, fp, fn = compute_panoptic_quality(gt_instances, pred_instances)
                metrics['loop'].append(loop)
                metrics['PQ'].append(PQ)
                metrics['SQ'].append(SQ)
                metrics['RQ'].append(RQ)
                metrics['Rec'].append(round(tp/(tp + fn), 2) if tp + fn > 0 else 0)
                metrics['Pre'].append(round(tp/(tp + fp),2) if tp + fp > 0 else 0)
        df_metrics = pd.DataFrame(metrics)
        pd.DataFrame(metrics).to_csv(os.path.join(data_folder, 'gt_metrics.csv'), sep=';', index=None)

    if compute_metrics == False:
        # load metrics
        df_metrics = pd.read_csv(os.path.join(data_folder, 'gt_metrics.csv'), sep=';')

    # plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        # average over all the samples
        df_data_metric = df_metrics[['loop', metric]]
        df_data_metric = df_data_metric[df_data_metric[metric] != 0]
        df_data_metric = df_data_metric.groupby("loop").mean()

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

if __name__ == '__main__':
    src_data_gt = r"D:\PDM_repo\Github\PDM\results\eval\20250622_224533_flatten_with_split_correction_long"
    # show_pseudo_labels_vs_gt(
    #     src_data_gt, 
    #     src_location=os.path.join(src_data_gt, "images/peudo_labels_vs_gt.png"), 
    #     compute_metrics=False,
    #     save_figure=True, 
    #     show_figure=True),

    src_data_train = r"D:\PDM_repo\Github\PDM\results\trainings_saved\20250621_221244_flatten_with_split_correction_long\training_metrics.csv"
    src_data_inf = r"D:\PDM_repo\Github\PDM\results\trainings_saved\20250621_221244_flatten_with_split_correction_long\inference_metrics.csv"
    src_data_semantic = r"D:\PDM_repo\Github\PDM\results\trainings_saved\20250621_221244_flatten_with_split_correction_long"

    show_stages_losses(src_data_train, src_location=os.path.join(src_data_semantic, "images/loss.png"), save_figure=False, show_figure=True)
    quit()
    show_training_losses(src_data_train, src_location=os.path.join(src_data_semantic, "images/losses.png"), save_figure=False, show_figure=True)
    show_pseudo_labels_evolution(src_data_semantic, src_location=os.path.join(src_data_semantic, "images/pseudo_labels_results.png"), save_figure=True, show_figure=False)
    show_global_metrics(src_data_train, src_location=os.path.join(src_data_semantic, "images/training_metrics.png"), save_figure=True, show_figure=False)
    show_inference_counts(src_data_inf, src_location=os.path.join(src_data_semantic, "images/inference_count.png"), save_figure=True, show_figure=False)
    show_problematic_empty(src_data_inf, src_location=os.path.join(src_data_semantic, "images/problematic_empty.png"), save_figure=True, show_figure=False)
    show_inference_metrics(src_data_inf, src_location=os.path.join(src_data_semantic, "images/inference_metrics.png"), save_figure=True, show_figure=False)