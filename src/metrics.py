import os
import numpy as np
import pandas as pd


def compute_classification_results(src_results):
    """
    Computes the number of instances for each class in the classification results.

    Args:
        - src_results (str): Path to the directory containing the classification results.

    Returns:
        - list: A list containing the count of instances for each class (0, 1, 2).
    """

    df_results = pd.read_csv(os.path.join(src_results, 'results.csv'), sep=';')
    vals = range(3)
    return [len(df_results[df_results['class'] == val]) for val in vals]


def compute_panoptic_quality(gt_instances, pred_instances):
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ) between ground truth and predicted instances.

    Args:
        - gt_instances (list): List of sets, each containing point indices for a ground truth instance.
        - pred_instances (list): List of sets, each containing point indices for a predicted instance.

    Returns:
        - tuple: PQ, SQ, RQ, tp (true positives), fp (false positives), fn (false negatives)
    """

    # convert inputs
    gt_instances, pred_instances = format_segmentation_for_PQ(gt_instances, pred_instances)

    tp, fp, fn = 0, 0, 0
    iou_sum = 0

    # Match predicted instances to ground truth instances
    matched_gt = set()
    matched_pred = set()
    
    for i, gt in enumerate(gt_instances):
        best_iou = 0
        best_pred = None

        for j, pred in enumerate(pred_instances):
            iou = len(gt & pred) / len(gt | pred)  # IoU computation
            
            if iou > best_iou:
                best_iou = iou
                best_pred = j
        
        # Threshold for a valid match
        if best_iou > 0.5:
            matched_gt.add(i)
            matched_pred.add(best_pred)
            tp += 1
            iou_sum += best_iou
        else:
            fn += 1  # Unmatched ground truth instance
    
    fp = len(pred_instances) - len(matched_pred)  # Unmatched predictions

    RQ = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) > 0 else 0
    SQ = iou_sum / tp if tp > 0 else 0
    PQ = SQ * RQ

    return PQ, SQ, RQ, tp, fp, fn


def compute_mean_iou(y_true, y_pred, num_classes=2):
    """
    Computes mean Intersection over Union (mIoU) for a set of ground truth and predicted labels.

    Args:
        - y_true (numpy.ndarray): Ground truth labels (N,).
        - y_pred (numpy.ndarray): Predicted labels (N,).
        - num_classes (int, optional): Total number of classes (default is 2).

    Returns:
        - float: Mean IoU score.
    """

    iou_list = []
    
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        iou_list.append(iou)

    return np.mean(iou_list)


def get_segmentation(instance_list, semantic_list):
    """
    Computes instance and semantic segmentation from lists of instance and semantic labels.

    Args:
        - instance_list (list): List of instance labels for each point.
        - semantic_list (list): List of semantic labels for each point.

    Returns:
        - tuple: Two lists: one containing sets of points for each instance, and the other for each semantic class.
    """

    instances_format = []
    semantic_format = []
    
    # Computing instances
    for instance in set(instance_list):
        if instance == 0: continue
        list_points = [pos for pos, val in enumerate(instance_list) if val == instance]
        instances_format.append(set(list_points))

    # Computing semantic
    for semantic in set(semantic_list):
        list_points = [pos for pos, val in enumerate(semantic_list) if val == semantic]
        semantic_format.append(set(list_points))

    return instances_format, semantic_format


def format_segmentation_for_PQ(instance_gt, instance_pred):
    """
    Formats the ground truth and predicted instance labels into sets for Panoptic Quality (PQ) calculation.

    Args:
        - instance_gt (list): List of ground truth instance labels for each point.
        - instance_pred (list): List of predicted instance labels for each point.

    Returns:
        - tuple: Two lists of sets: one for ground truth instances, and one for predicted instances.
    """

    gt_format = []
    preds_format = []
    
    # Computing instances
    for instance in set(instance_gt):
        if instance == 0: continue
        list_points = [pos for pos, val in enumerate(instance_gt) if val == instance]
        gt_format.append(set(list_points))

    for instance in set(instance_pred):
        if instance == 0: continue
        list_points = [pos for pos, val in enumerate(instance_pred) if val == instance]
        preds_format.append(set(list_points))

    return gt_format, preds_format


if __name__ == '__main__':
    src = r"D:\PDM_repo\Github\PDM\data\dataset_tiles_100m\temp\loops\0\preds\color_grp_full_tile_100_out_split_instance\results"
    src2 = r"D:\PDM_repo\Github\PDM\results\trainings\20250415_152844_test\inference_metrics.csv"
    [num_garbage, num_multi, num_single] = compute_classification_results(src)
    print(num_garbage, '-', num_multi, '-', num_single)
    df = pd.read_csv(src2, sep=';')
    file="color_grp_full_tile_1.laz"
    loop=0
    print(df.columns)
    df.loc[(df.name == file) & (df.num_loop == loop), "num_garbage"] = num_garbage
    df.loc[(df.name == file) & (df.num_loop == loop), "num_multi"] = num_multi
    df.loc[(df.name == file) & (df.num_loop == loop), "num_single"] = num_single
    df.loc[(df.name == file) & (df.num_loop == loop), "num_predictions"] = num_garbage + num_multi + num_single
    df.to_csv(
        src2,
        sep=';',
        index=False,
    )
