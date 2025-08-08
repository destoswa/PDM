import os
import sys
import numpy as np
import laspy
import pickle
import torch
from time import time

if __name__ == "__main__":
    sys.path.append("D:/PDM_repo/Github/PDM")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from models.KDE_classifier.models.model import KDE_cls_model
from models.KDE_classifier.src.utils import ToKDE
from omegaconf import OmegaConf
from packaging import version



def fast_inference(samples, args):
    SRC_MODEL = "../models/KDE_classifier/models/pretrained/model_KDE.tar"
    """
    Perform fast inference on samples using a pre-trained KDE-based model.

    Parameters:
    - samples (list): List of input samples to infer on.
    - args (dict): Dictionary containing inference parameters.

    Returns:
    - np.ndarray: Model predictions as a numpy array.
    """
    
    conf = {
        "num_class": args['inference']['num_class'],
        "grid_dim": args['inference']['grid_size'],
    }

    # Load the model
    model = KDE_cls_model(conf).to(torch.device('cuda'))
    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        checkpoint = torch.load("../models/KDE_classifier/models/pretrained/model_KDE.tar", weights_only=False)
    else:
        checkpoint = torch.load("../models/KDE_classifier/models/pretrained/model_KDE.tar")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create the transform object
    kde_transform = ToKDE(
        args['inference']['grid_size'], 
        args['inference']['kernel_size'], 
        args['inference']['num_repeat_kernel'],
        )

    lst_grids = []

    # Prepare samples for the model (KDE transformation)
    for sample in samples:
        data = {'data': sample, 'label': 0}
        sample_kde = kde_transform(data)
        grid, _ = sample_kde['data'], sample_kde['label']
        grid = grid.cuda()
        lst_grids.append(grid)
    
    # Prepare batch
    if len(lst_grids) == 1:
        batch = grid.reshape(1,grid.shape[0],grid.shape[1],grid.shape[2])
    else:
        batch = torch.stack(lst_grids, axis=0)
    batch = batch.cuda()

    # Make predictions
    pred = model(batch)

    return np.array(pred.detach().cpu())


if __name__ == "__main__":
    args = OmegaConf.load(r"D:\PDM_repo\Github\PDM\models\KDE_classifier\config\inference.yaml")
    src_tile = r"D:\PDM_repo\Github\PDM\results\samples_split_fail\tile_3.laz"
    src_mask = r"D:\PDM_repo\Github\PDM\results\samples_split_fail\mask3.pickle"
    src_other_mask = r"D:\PDM_repo\Github\PDM\results\samples_split_fail\other_mask3.pickle"
    tile = laspy.read(src_tile)
    with open(src_mask, 'rb') as file:
        mask = pickle.load(file)
    with open(src_other_mask, 'rb') as file:
        other_mask = pickle.load(file)
    coordx = np.array(getattr(tile, 'x'))
    coordy = np.array(getattr(tile, 'y'))
    coordz = np.array(getattr(tile, 'z'))
    pointCloud = np.stack([coordx, coordy, coordz], axis=1)
    sample = pointCloud[mask]
    other_sample = pointCloud[other_mask]
    start = time()
    preds = fast_inference([sample, other_sample], args)
    print(time() - start)
    # test = np.array([2,2])
    if np.any(np.argmax(preds, axis=1) != 2):
        print("nope!")
    print(np.argmax(preds, axis=1))