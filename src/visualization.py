import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import laspy
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

if __name__ == "__main__":
    from metrics import compute_panoptic_quality
else:
    from src.metrics import compute_panoptic_quality


def show_metric_over_epoch(df, metric_name, ax=None, save_figure=False, src_figure=None, show_figure=False):
    """
    Plots the evolution of a given metric over epochs, separated by training stage.

    Args:
        - df (pd.DataFrame): Dataframe containing the metrics and training progress information.
        - metric_name (str): Name of the metric to plot.
        - ax (plt.Axes, optional): Matplotlib axis to plot on, if provided.
        - save_figure (bool, optional): If True, saves the figure to disk.
        - src_figure (str, optional): Path to save the figure, if saving is enabled.
        - show_figure (bool, optional): If True, displays the figure.

    Returns:
        - plt.Figure: The generated matplotlib figure.
    """

    fig = None
    if ax == None:
        fig, ax = plt.figure()
    
    for stage in df.stage.unique():
        df_stage = df[df.stage == stage]
        ax.plot(np.array(df_stage.num_epoch), np.array(df_stage[metric_name]), label=stage)

    if save_figure:
        if src_figure != None and fig != None:
            plt.savefig(src_figure)
            plt.savefig(src_figure.split('.png')[0] + '.eps', format='eps')
        else:
            raise UserWarning("When saving figure, the ax should not be precised and the src should be precise!")
        
    if show_figure and fig != None:
        plt.show()
        plt.close()


def show_metric_over_samples(df, metric_name, ax=None, save_figure=False, src_figure=None, show_figure=False):
    """
    Plots the evolution of a given metric over samples (index-based), useful for inspecting batch/sample-wise behavior.

    Args:
        - df (pd.DataFrame): Dataframe containing the metric values across samples.
        - metric_name (str): Name of the metric to plot.
        - ax (plt.Axes, optional): Matplotlib axis to plot on, if provided.
        - save_figure (bool, optional): If True, saves the figure to disk.
        - src_figure (str, optional): Path to save the figure, if saving is enabled.
        - show_figure (bool, optional): If True, displays the figure.

    Returns:
        - plt.Figure: The generated matplotlib figure.
    """

    fig = None
    if ax == None:
        fig, ax = plt.figure()
    x = np.array(df.index)
    y = np.array(df[metric_name])
    ax.plot(x, y)
    ax.set_ylim([max(0, np.min(y) - 0.05), min(np.max(y) + 0.05, 100)])

    if save_figure:
        if src_figure != None and fig != None:
            plt.savefig(src_figure)
            plt.savefig(src_figure.split('.png')[0] + '.eps', format='eps')
        else:
            raise UserWarning("When saving figure, the ax should not be precised and the src should be precise!")
        
    if show_figure and fig != None:
        plt.show()
        plt.close()


def show_global_metrics(data_src, exclude_columns = ['num_loop', 'num_epoch', 'stage', 'map'], src_location=None, show_figure=True, save_figure=False):
    """
    Plots all global metrics from a CSV file over training epochs in a grid layout.

    Args:
        - data_src (str or pd.DataFrame): Path to CSV file or preloaded DataFrame with metrics.
        - exclude_columns (list, optional): List of columns to exclude from plotting.
        - src_location (str, optional): Path to save the figure, if saving is enabled.
        - show_figure (bool, optional): If True, displays the figure.
        - save_figure (bool, optional): If True, saves the figure to disk.

    Returns:
        - plt.Figure: The generated matplotlib figure with global metrics.
    """

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
        plt.savefig(src_location.split('.png')[0] + '.eps', format='eps')

    if show_figure:
        plt.show()


def show_training_losses(data_src, src_location=None, show_figure=True, save_figure=False):
    """
    Plots the overall training loss and its sub-losses over epochs.

    Args:
        - data_src (str or pd.DataFrame): Path to CSV file or preloaded DataFrame with loss values.
        - src_location (str, optional): Path to save the figure, if saving is enabled.
        - show_figure (bool, optional): If True, displays the figure.
        - save_figure (bool, optional): If True, saves the figure to disk.

    Returns:
        - plt.Figure: The generated matplotlib figure showing loss evolution.
    """
    
    # load and prepare data
    df_data = pd.read_csv(data_src, sep=';')
    df_data = df_data.loc[df_data.stage == 'train']
    sublosses_names = ['offset_norm_loss', 'offset_dir_loss', 'ins_loss', 'ins_var_loss', 'ins_dist_loss', 'ins_reg_loss', 'semantic_loss', 'score_loss']

    # plot
    fig = plt.figure(figsize=(8,5))
    plt.plot(np.array(df_data.num_epoch), np.array(df_data.loss), label='loss', linewidth=2.5)
    for loss_name in sublosses_names:
        plt.plot(np.array(df_data.num_epoch), np.array(df_data[loss_name]), label=loss_name, linewidth=1.3, alpha=0.3)
    plt.xlabel("Epoch [-]")
    plt.ylabel("Loss value [-]")
    plt.title("Evolution of losses by SegmentAnyTree")
    plt.legend(ncol=3)

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)
        plt.savefig(src_location.split('.png')[0] + '.eps', format='eps')

    if show_figure:
        plt.show()


def show_stages_losses(data_src, exclude_columns = ['num_loop', 'num_epoch', 'stage', 'map'], src_location=None, show_figure=True, save_figure=False):
    """
    Plots the smoothed loss curves over epochs for different training stages (e.g., train, validation).

    Args:
        - data_src (str or pd.DataFrame): Path to CSV file or preloaded DataFrame with loss values.
        - exclude_columns (list, optional): Columns to exclude from plotting.
        - src_location (str, optional): Path to save the figure, if saving is enabled.
        - show_figure (bool, optional): If True, displays the figure.
        - save_figure (bool, optional): If True, saves the figure to disk.

    Returns:
        - plt.Figure: The generated matplotlib figure with stage-wise losses.
    """
    
    # load and prepare data
    df_data = pd.read_csv(data_src, sep=';')
    stages = list(df_data.stage.unique())
    stages.remove('test')
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ['Training loss', 'Validation loss']

    # plot
    fig = plt.figure(figsize=(8,5))
    for id_stage, stage in enumerate(stages):
        df_subdata = df_data.loc[df_data.stage == stage]
        plt.plot(np.array(df_subdata.num_epoch), np.array(df_subdata.loss), linewidth=1.5, alpha=0.3, color=colors[id_stage])

        # smoothing
        n = 5
        num_rep = 5
        y_smooth = uniform_filter1d(np.array(df_subdata.loss), size=n)
        for i in range(num_rep - 1):
            y_smooth = uniform_filter1d(y_smooth, size=n)
        plt.plot(np.array(df_subdata.num_epoch), y_smooth, label=labels[id_stage], linewidth=2.5, color=colors[id_stage])
    plt.xlabel("Epoch [-]")
    plt.ylabel("Loss value [-]")
    plt.title("Evolution of losses per set")
    plt.legend()

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)
        plt.savefig(src_location.split('.png')[0] + '.eps', format='eps')

    if show_figure:
        plt.show()


def show_inference_counts(data_src, src_location=None, show_figure=True, save_figure=False):
    """
    Plots counts and fractions of different prediction types (e.g., empty, problematic) over training loops.

    Args:
        - data_src (str or pd.DataFrame): Path to CSV file or preloaded DataFrame with inference counts.
        - src_location (str, optional): Path to save the figure, if saving is enabled.
        - show_figure (bool, optional): If True, displays the figure.
        - save_figure (bool, optional): If True, saves the figure to disk.

    Returns:
        - plt.Figure: The generated matplotlib figure with inference counts.
    """
    
    df_data = pd.read_csv(data_src, sep=';')
    averages = df_data[["num_loop", "num_predictions", "num_garbage", "num_multi", "num_single"]].groupby('num_loop').mean()
    fractions = averages[["num_garbage", "num_multi", "num_single"]].div(averages["num_predictions"], axis=0)

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs = axs.flatten()
    for i, data in enumerate([averages.drop('num_predictions', axis=1), fractions]):
        df = pd.DataFrame(data)
        for col in df.columns:
            axs[i].plot(np.array(df.index), np.array(df[col]), label=col)
            axs[i].legend()
    axs[0].set_title('Count of the differente types of predictions')
    axs[1].set_title('Fraction over number of predictions')
    
    # set limits
    axs[0].set_ylim([0,np.max(averages.drop('num_predictions', axis=1).max().values)*1.1])
    axs[1].set_ylim([0,1])

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)
        plt.savefig(src_location.split('.png')[0] + '.eps', format='eps')

    if show_figure:
        plt.show()


def show_problematic_empty(data_src, src_location=None, show_figure=True, save_figure=False):
    """
    Plots the number of problematic and empty samples across training loops.

    Args:
        - data_src (str or pd.DataFrame): Path to CSV file or preloaded DataFrame with sample information.
        - src_location (str, optional): Path to save the figure, if saving is enabled.
        - show_figure (bool, optional): If True, displays the figure.
        - save_figure (bool, optional): If True, saves the figure to disk.

    Returns:
        - plt.Figure: The generated matplotlib figure showing problematic/empty counts.
    """
    
    df_data = pd.read_csv(data_src, sep=';')
    num_problematic = df_data[['num_loop', 'is_problematic']].groupby('num_loop').sum()
    num_empty = df_data[['num_loop', 'is_empty']].groupby('num_loop').sum()

    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs = axs.flatten()
    for i, data in enumerate([num_problematic, num_empty]):
        df = pd.DataFrame(data)
        for col in df.columns:
            axs[i].plot(np.array(df.index), np.array(df[col]), label=col)
            axs[i].legend()
    axs[0].set_title('Number of problematic samples')
    axs[1].set_title('Number of empty samples')

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)
        plt.savefig(src_location.split('.png')[0] + '.eps', format='eps')

    if show_figure:
        plt.show()


def show_inference_metrics(data_src, metrics = ['PQ', 'SQ', 'RQ', 'Pre', 'Rec'], src_location=None, show_figure=True, save_figure=False):
    """
    Plots inference metrics (e.g., PQ, SQ, RQ, Precision, Recall) averaged over samples across training loops.

    Args:
        - data_src (str or pd.DataFrame): Path to CSV file or preloaded DataFrame with inference metrics.
        - metrics (list, optional): List of metrics to plot.
        - src_location (str, optional): Path to save the figure, if saving is enabled.
        - show_figure (bool, optional): If True, displays the figure.
        - save_figure (bool, optional): If True, saves the figure to disk.

    Returns:
        - plt.Figure: The generated matplotlib figure showing inference metrics.
    """

    abrev_to_name = {
        'PQ': "Panoptic Quality",
        'SQ': "Segmentation Quality",
        'RQ': "Recognition Quality",
        'Pre': "Precision",
        'Rec': "Recall",
    }
    df_data = pd.read_csv(data_src, sep=';')
    df_data = df_data.loc[df_data.num_loop != 0]

    num_rows = int(np.ceil(len(metrics)/2))
    num_cols = min(len(metrics), 2)

    # plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 4 * num_rows), sharex=True, sharey=False)
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
        plt.savefig(src_location.split('.png')[0] + '.eps', format='eps')

    if show_figure:
        plt.show()


def show_pseudo_labels_evolution(data_folder, src_location=None, only_fancy_inst_count=False, do_per_cluster=False, cluster_csv_file=None, show_figure=True, save_figure=False):
    """
    Plots the evolution of pseudo-label characteristics (instance count, stability, etc.) over training loops, optionally per cluster.

    Args:
        - data_folder (str): Path to the folder containing pseudo-label CSV files.
        - src_location (str, optional): Path to save the figure, if saving is enabled.
        - only_fancy_inst_count (bool, optional): If True, plots only the 'fancy instance count'.
        - do_per_cluster (bool, optional): If True, performs analysis per cluster based on cluster CSV.
        - cluster_csv_file (str, optional): Path to cluster mapping CSV file if per-cluster analysis is enabled.
        - show_figure (bool, optional): If True, displays the figure.
        - save_figure (bool, optional): If True, saves the figure to disk.

    Returns:
        - plt.Figure: The generated matplotlib figure showing pseudo-label evolution.
    """
    
    # load and generate data to show
    num_loop = 0
    count_sem = {}
    count_inf = {}
    change_from_previous = {}
    total_not_change = {}
    not_change_in_tile = {}
    previous_tiles = {}
    per_cluster = {}

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

    if do_per_cluster:
        df_clusters = pd.read_csv(cluster_csv_file, sep=';')
        clusters = df_clusters.cluster_id.unique()
        per_cluster = {x: [] for x in clusters}
        for _, cluster in tqdm(enumerate(clusters), total=len(clusters), desc="Processing pseudo-labels for visualization"):
            count_inf = {}

            lst_tiles = list(df_clusters.loc[df_clusters.cluster_id == cluster].tile_name)
            removed_tiles = ["color_grp_full_tile_10.laz", "color_grp_full_tile_385.laz"]
            for tile in removed_tiles:
                if tile in lst_tiles:
                    lst_tiles.remove(tile)

            # processing each loop
            for num_loop in lst_loops:
                count_inf[num_loop] = []
                for tile_src in [os.path.join(data_folder, str(num_loop), "pseudo_labels", x) for x in lst_tiles]:
                    if not os.path.exists(tile_src):
                        continue
                    tile = laspy.read(os.path.join(data_folder, str(num_loop), "pseudo_labels", tile_src))
                    count_inf[num_loop].append(len(set(tile.treeID)))

            # aggregation
            count_inf_agg = []
            count_inf_std = []
            for num_loop in count_inf.keys():
                count_inf_agg.append(np.mean(list(count_inf[num_loop])))
                count_inf_std.append(np.std(list(count_inf[num_loop])))
            per_cluster[cluster].append(count_inf_agg)
            per_cluster[cluster].append(count_inf_std)
    else:
        # processing each loop
        for _, num_loop in tqdm(enumerate(lst_loops), total=len(lst_loops), desc="Processing pseudo-labels for visualization"):
            count_sem[num_loop] = {}
            change_from_previous[num_loop] = {}
            total_not_change[num_loop] = {}
            not_change_in_tile[num_loop] = {}
            
            lst_tiles = os.listdir(os.path.join(data_folder, str(num_loop), "pseudo_labels"))
            removed_tiles = ["color_grp_full_tile_10.laz", "color_grp_full_tile_385.laz"]
            for tile in removed_tiles:
                if tile in lst_tiles:
                    lst_tiles.remove(tile)

            count_inf[num_loop] = []
            for tile_src in lst_tiles:
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
        count_inf_std = []
        change_from_previous_agg = {x: [] for x in categories}
        total_not_change_agg = {x: [] for x in categories}
        for num_loop in count_sem.keys():
            count_inf_agg.append(np.mean(list(count_inf[num_loop])))
            count_inf_std.append(np.std(list(count_inf[num_loop])))
            for id_cat, cat in enumerate(categories):
                count_sem_agg[cat].append(np.mean([tile_val for tile_val in count_sem[num_loop].values()], axis=0)[id_cat])
                if num_loop > 0:
                    change_from_previous_agg[cat].append(np.mean([tile_val for tile_val in change_from_previous[num_loop].values()], axis=0)[id_cat])
                total_not_change_agg[cat].append(np.mean([tile_val for tile_val in total_not_change[num_loop].values()], axis=0)[id_cat])
    
    # visualizing
    if only_fancy_inst_count:
        if do_per_cluster:
            lst_text_y_add = [2, 2, -8, 2]
            lst_clusters_labels = ['Crowded + Flat', 'Crowded + Steep', 'Empty + Steep', 'Empty + Flat']
            fig = plt.figure(figsize=(12,8))
            for id_cluster, (cluster, data) in enumerate(per_cluster.items()):
                # print(data[0])
                count_inf_agg = data[0]
                x = np.array(range(len(count_inf_agg)))
                y = np.array(count_inf_agg)
                span = (np.max(y)+5) - (np.min(y)-5)
                sns.lineplot(x=x, y=y, errorbar=None, marker='o', label=lst_clusters_labels[id_cluster])

                # Annotate each point
                for i in range(len(x)):
                    plt.text(x[i], y[i] + lst_text_y_add[id_cluster],  # slightly above the error bar
                            f"{round(y[i],2)}",              # formatted value
                            ha='center', va='bottom', fontsize=10, color='black')
            plt.title('Average number of instances in tiles')
            plt.xlabel('Loop [-]')
            plt.ylabel('Number of instances [-]')
            plt.legend()

            if save_figure and src_location != None:
                print()
                plt.savefig(src_location.split('.png')[0] + '_count_instances_per_cluster.png')
                plt.savefig(src_location.split('.png')[0] + '_count_instances_per_cluster.eps', format='eps')

            if show_figure:
                plt.show()
        else:
            fig = plt.figure(figsize=(6,4))
            x = np.array(range(len(count_inf_agg)))
            y = np.array(count_inf_agg)
            span = (np.max(y)+5) - (np.min(y)-5)
            sns.lineplot(x=x, y=y, errorbar=None, marker='o', label=None)

            # Annotate each point
            for i in range(len(x)):
                plt.text(x[i], y[i] + 0.8 * span / 30,  # slightly above the error bar
                        f"{round(y[i],2)}",              # formatted value
                        ha='center', va='bottom', fontsize=10, color='black')
            plt.ylim((np.min(y)-5, np.max(y)+5))
            plt.title('Average number of instances in tiles')
            plt.xlabel('Loop [-]')
            plt.ylabel('Number of instances [-]')
            
            plt.tight_layout()

            # save csv
            pd.DataFrame({
                "loop": x,
                "num_preds": y
            }).to_csv(os.path.join(data_folder, 'pseudo_labels_num_instances.csv'), sep=';', index=False)

            if save_figure and src_location != None:
                plt.savefig(src_location.split('.png')[0] + '_count_instances.png')
                plt.savefig(src_location.split('.png')[0] + '_count_instances.eps', format='eps')

            if show_figure:
                plt.show()

    else:
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
            plt.savefig(src_location.split('.')[0] + '.eps', format='eps')

        if show_figure:
            plt.show()
    

def show_pseudo_labels_vs_gt(data_folder, src_location=None, metrics = ['PQ', 'SQ', 'RQ', 'Pre', 'Rec'], compute_metrics=True, show_figure=True, save_figure=False):
    """
    Visualizes the evolution of pseudo-label quality metrics (PQ, SQ, RQ, Precision, Recall) across self-training loops.

    Args:
        - data_folder (str): Path to the folder containing loop subfolders with pseudo-labels and optionally a precomputed metrics CSV.
        - src_location (str, optional): File path to save the figure. Default is None.
        - metrics (list, optional): List of metrics to visualize. Default is ['PQ', 'SQ', 'RQ', 'Pre', 'Rec'].
        - compute_metrics (bool, optional): If True, computes metrics from pseudo-labels; otherwise loads from CSV. Default is False.
        - show_figure (bool, optional): If True, displays the figure. Default is True.
        - save_figure (bool, optional): If True, saves the figure to src_location. Default is False.

    Returns:
        - None
    """

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
        df_data_metric = df_data_metric.groupby(df_data_metric["loop"]).mean()

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
        plt.savefig(src_location.split('.')[0] + '.eps', format='eps')

    if show_figure:
        plt.show()


def show_grid_search_metric(target_src, data, params, title, do_save=True, do_show=False):
    """
    Plots a heatmap of a metric resulting from a grid search over two hyperparameters.

    Args:
    - target_src (str): File path where the plot will be saved.
    - data (DataFrame): DataFrame containing grid search results.
    - params (list): the name of the columns corresponding to the two params in data.
    - title (str): Title of the plot.
    - do_save (bool, optional): If True, save the plot. Default is True.
    - do_show (bool, optional): If True, display the plot. Default is False.

    Returns:
    - None
    """
    fig = plt.figure(figsize=(8,6))
    glue = data.pivot(index=params[0], columns=params[1], values="value")
    sns.heatmap(glue, annot=True, cmap=sns.color_palette("Blues", as_cmap=True), fmt='g')
    plt.title(title)
    plt.xlabel(params[1])
    plt.ylabel(params[0])

    if do_save:
        plt.savefig(target_src)
        plt.savefig(target_src.split('.')[0] + '.eps', format='eps')

    if do_show:
        plt.show()

    plt.close()


def show_grid_search(data_folder,  name_params = None, show_figure=True, save_figure=False):
    """
    Processes and visualizes grid search results, showing number of instances and multiple quality metrics.

    Args:
        - data_folder (str): Path to the folder containing grid search results and subfolders.
        - name_params (list, optional): List of two parameter names (columns in grid_search.csv). Default is None.
        - show_figure (bool, optional): If True, displays the plots. Default is True.
        - save_figure (bool, optional): If True, saves the plots in the 'images' subfolder. Default is False.

    Returns:
        - None
    """

    os.makedirs(os.path.join(data_folder, 'images'), exist_ok=True)
    df_links = pd.read_csv(os.path.join(data_folder, 'grid_search.csv'), sep=';')

    # show num predictions
    max_num_instances = []
    for link in df_links.src:
        src_link = os.path.join(data_folder, link)
        if not os.path.exists(os.path.join(src_link, 'pseudo_labels_num_instances.csv')):
            print(f"computing num instances for {link}...")
            show_pseudo_labels_evolution(src_link, save_figure=False, show_figure=False, only_fancy_inst_count=True)
        df_data = pd.read_csv(os.path.join(src_link, 'pseudo_labels_num_instances.csv'), sep=';')
        max_num_instances.append(df_data.num_preds.max())
    df_links.insert(1, "value", max_num_instances)
    show_grid_search_metric(os.path.join(data_folder, 'images', 'grid_search_num_instances.png'), df_links, ['num_epochs', 'num_samples'], 'Grid Search - Num predictions', do_save=save_figure, do_show=show_figure)

    # show inference metrics
    lst_metrics = ["PQ", "SQ", "RQ", "Pre", "Rec"]
    lst_titles = ["Panoptic Quality", "Semantic Quality", "Recognition Quality", "Precision", "Recall"]
    for id_metric, metric in enumerate(lst_metrics):
        max_metric = []
        for link in df_links.src:
            src_link = os.path.join(data_folder, link)
            df_data = pd.read_csv(os.path.join(src_link, 'inference_metrics.csv'), sep=';')
            df_data = df_data.drop(df_data[df_data['num_loop'] == 0].index)
            max_metric.append(list(df_data[metric])[-1])
        df_links.value = max_metric
        show_grid_search_metric(os.path.join(data_folder, 'images', f'grid_search_{lst_titles[id_metric]}.png'), df_links, ['num_epochs', 'num_samples'], f'Grid Search - {lst_titles[id_metric]}', do_save=save_figure, do_show=show_figure)


def show_recall_precision_per_cluster(data_src, metrics = ['Pre', 'Rec'], src_location=None, cluster_csv_file=None, show_figure=True, save_figure=False):
    """
    Plots Precision and Recall per cluster of tiles across training loops.

    Args:
        - data_src (str): Path to CSV file containing loop-wise metrics per tile.
        - metrics (list, optional): List of metrics to plot (default ['Pre', 'Rec']).
        - src_location (str, optional): Path to save the figure. Default is None.
        - cluster_csv_file (str, optional): Path to CSV file containing cluster assignments. Default is None.
        - show_figure (bool, optional): If True, displays the figure. Default is True.
        - save_figure (bool, optional): If True, saves the figure to src_location. Default is False.

    Returns:
        - None
    """
    
    lst_clusters_labels = ['Crowded + Flat', 'Crowded + Steep', 'Empty + Steep', 'Empty + Flat']
    df_clusters = pd.read_csv(cluster_csv_file, sep=';')
    clusters = df_clusters.cluster_id.unique()
    for _, cluster in tqdm(enumerate(clusters), total=len(clusters), desc="Processing pseudo-labels for visualization"):
        pass
    abrev_to_name = {
        'Pre': "Precision",
        'Rec': "Recall",
    }
    df_data = pd.read_csv(data_src, sep=';')
    df_data = df_data.loc[df_data.num_loop != 0]

    # removing failing tiles
    removed_tiles = ["color_grp_full_tile_10.laz", "color_grp_full_tile_385.laz"]
    df_data = df_data.loc[~df_data.name.isin(removed_tiles)]

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        y_min = 100
        y_max = 0 
        for id_cluster, cluster in enumerate(clusters):
            lst_tiles = list(df_clusters.loc[df_clusters.cluster_id == cluster].tile_name)
            df_data_metric = df_data.loc[df_data["name"].isin(lst_tiles)]
            df_data_metric = df_data_metric[['num_loop', metric]]
            df_data_metric = df_data_metric[df_data_metric[metric] != 0]
            df_data_metric = df_data_metric.groupby("num_loop").mean()
            
            x = np.array(df_data_metric.index)
            y = np.array(df_data_metric[metric])
            y_min = np.min(y) if y_min > np.min(y) else y_min
            y_max = np.max(y) if y_max < np.max(y) else y_max
            axes[i].plot(x, y, label=lst_clusters_labels[id_cluster])

        axes[i].set_ylim([max(0, y_min - 0.05), min(y_max + 0.05, 100)])    
        axes[i].set_title(abrev_to_name[metric])
        axes[i].set_ylabel('Value [-]')
        axes[i].set_xlabel('Loops [-]')
        axes[i].legend()

    plt.tight_layout()
    if save_figure and src_location != None:
        plt.savefig(src_location)
        plt.savefig(src_location.split('.')[0] + '.eps', format='eps')

    if show_figure:
        plt.show()


def show_test_set(data_folder, src_location=None, cluster_csv_file=None, show_figure=True, save_figure=False):
    """
    Analyzes and visualizes the distribution of prediction classes (garbage, multi, single) per cluster over training loops.

    Args:
        - data_folder (str): Path to the folder containing loop prediction results.
        - src_location (str, optional): Path to save the figure. Default is None.
        - cluster_csv_file (str, optional): Path to CSV file containing cluster assignments for the test set. Default is None.
        - show_figure (bool, optional): If True, displays the figure. Default is True.
        - save_figure (bool, optional): If True, saves the figure to src_location. Default is False.

    Returns:
        - None
    """
    
    # finding the number of loops
    lst_loops = []
    num_loop = 0
    while True:
        if not str(num_loop) in os.listdir(os.path.join(data_folder, 'loops')):
            break
        lst_loops.append(num_loop)
        num_loop += 1
    if num_loop == 0:
        print("No loop folder from which to extract the pseudo-labels")
        quit()   

    # process evolution per cluster
    df_clusters = pd.read_csv(cluster_csv_file, sep=';')
    lst_tiles_tot = list(df_clusters.tile_name) 
    results_tot = {y: {x:{'garbage': [], 'multi': [], 'single': []} for x in lst_loops} for y in range(len(lst_tiles_tot))}

    lst_titles = ['Crowded + Flat', 'Crowded + Steep', 'Empty + Steep', 'Empty + Flat']
    lst_groups = df_clusters.cluster_id.unique()
    for id_group, group in enumerate(lst_groups):

        df_group = df_clusters.loc[df_clusters.cluster_id == group]

        lst_tiles = list(df_group.tile_name)
        
        for loop in lst_loops:
            src_evaluation = os.path.join(data_folder, "loops", str(loop), 'preds')
            lst_folders = [x for x in os.listdir(src_evaluation) if os.path.isdir(os.path.join(src_evaluation, x)) and x.split('_out_split_instance')[0]+'.laz' in lst_tiles]
            for folder in lst_folders:
                results_loop = pd.read_csv(os.path.join(src_evaluation, folder, "results/results.csv"), sep=';')
                
                for cat_num, cat_name in enumerate(['garbage', 'multi', 'single']):
                    results_tot[id_group][loop][cat_name].append(len(results_loop.loc[results_loop['class'] == cat_num]))

    results_agg = {
        x: {
            'garbage': [np.nanmean(results_tot[x][loop]["garbage"]) if len(results_tot[x][loop]["garbage"]) > 0 else 0 for loop in lst_loops],
            'multi': [np.nanmean(results_tot[x][loop]["multi"]) if len(results_tot[x][loop]["garbage"]) > 0 else 0 for loop in lst_loops],
            'single': [np.nanmean(results_tot[x][loop]["single"]) if len(results_tot[x][loop]["garbage"]) > 0 else 0 for loop in lst_loops],
            } for x in range(len(lst_tiles_tot))}

    fig, axs = plt.subplots(2,2,figsize=(12,12))
    axs = axs.flatten()
    for id_ax, ax in enumerate(axs):
        x = np.arange(len(lst_loops))
        sns.lineplot(x=x, y=np.array(results_agg[id_ax]['garbage']), ax=ax, label='garbage')
        sns.lineplot(x=x, y=np.array(results_agg[id_ax]['multi']), ax=ax, label='multi')
        sns.lineplot(x=x, y=np.array(results_agg[id_ax]['single']), ax=ax, label='single')

        ax.legend()
        ax.set_title(lst_titles[id_ax])

    if save_figure and src_location != None:
        plt.savefig(src_location)
        plt.savefig(src_location.split('.')[0] + '.eps', format='eps')

    if show_figure:
        plt.show()


if __name__ == '__main__':
    """
    In this main, you can run the different functions to produce the figures you want.
    The way it works is very basic. Comment every sections before the one you want to run and then comment the functions that you don't want to run.
    Every section ends with a "quit()" command so no need to comment what comes after
    """

    # # 1) To produce results related to the training of a pipeline
    # src_data_semantic = r"D:\PDM_repo\Github\PDM\results\for_paper\learning_rate\lr=0.001"    # only change this path
    # src_data_train = os.path.join(src_data_semantic, "training_metrics.csv")
    # src_data_inf = os.path.join(src_data_semantic, "inference_metrics.csv")
    # os.makedirs(os.path.join(src_data_semantic, 'images'), exist_ok=True)
    # # show_pseudo_labels_evolution(src_data_semantic, src_location=os.path.join(src_data_semantic, "images/pseudo_labels_results.png"), only_fancy_inst_count=True, save_figure=True, show_figure=False)
    # # show_pseudo_labels_evolution(src_data_semantic, src_location=os.path.join(src_data_semantic, "images/pseudo_labels_results.png"), save_figure=True, show_figure=False)
    # # show_global_metrics(src_data_train, src_location=os.path.join(src_data_semantic, "images/training_metrics.png"), save_figure=True, show_figure=False)
    # # show_inference_counts(src_data_inf, src_location=os.path.join(src_data_semantic, "images/inference_count.png"), save_figure=True, show_figure=False)
    # # quit()
    # # show_problematic_empty(src_data_inf, src_location=os.path.join(src_data_semantic, "images/problematic_empty.png"), save_figure=True, show_figure=False)
    # # show_inference_metrics(src_data_inf, src_location=os.path.join(src_data_semantic, "images/inference_metrics.png"), save_figure=True, show_figure=False)
    # # show_inference_metrics(src_data_inf, metrics = ['Pre', 'Rec'], src_location=os.path.join(src_data_semantic, "images/inference_Rec_Pre.png"), save_figure=True, show_figure=False)
    # show_stages_losses(src_data_train, src_location=os.path.join(src_data_semantic, "images/loss.png"), save_figure=True, show_figure=False)
    # # show_training_losses(src_data_train, src_location=os.path.join(src_data_semantic, "images/losses.png"), save_figure=True, show_figure=False)
    # quit()


    # # # 2) To produce results related to a training-set with groups (called clustered here)
    # # src_data_semantic = r"D:\PDM_repo\Github\PDM\results\for_paper\grid_search\20250627_205749_gs_3_500"
    # # src_clusters = r"D:\PDM_repo\Github\PDM\results\for_paper\final\final_training\training_set.csv"
    # # show_pseudo_labels_evolution(src_data_semantic, src_location=os.path.join(src_data_semantic, "images/pseudo_labels_results.png"), do_per_cluster=True, cluster_csv_file=src_clusters, only_fancy_inst_count=True, save_figure=True, show_figure=True)
    # # show_recall_precision_per_cluster(src_data_inf, src_location=os.path.join(src_data_semantic, "images/recall_precission_per_cluster.png"), cluster_csv_file=src_clusters, show_figure=False, save_figure=True)
    # # quit()


    # 3) To produce results related to a test-set with groups (called clustered here)
    src_clusters_test = r"D:\PDM_repo\Github\PDM\data\final_dataset\testing_set.csv"
    src_data_test_set = r"D:\PDM_repo\Github\PDM\results\eval\20250729_181022_inf_group_final"
    show_test_set(src_data_test_set, src_location=os.path.join(src_data_test_set, 'images/test_set_results.png'), cluster_csv_file=src_clusters_test, show_figure=True, save_figure=True)
    quit()


    # # 4) To produce results related to ground truth
    # src_data_gt = r"D:\PDM_repo\Github\PDM\results\eval\20250701_162429_final_on_gt"
    # show_pseudo_labels_vs_gt(src_data_gt, src_location=os.path.join(src_data_gt, "images/peudo_labels_vs_gt.png"), compute_metrics=True, save_figure=True, show_figure=True)
    # quit()


    # 5) To produce results related to a grid search
    src_grid_search = r"D:\PDM_repo\Github\PDM\results\for_paper\grid_search"
    show_grid_search(src_grid_search, name_params=['Num epochs per loop', 'Num samples per epoch'], save_figure=True, show_figure=False)
    quit()
