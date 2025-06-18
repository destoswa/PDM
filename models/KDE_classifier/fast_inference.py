import os
import pandas as pd
import shutil

from tqdm import tqdm
import laspy
import pickle
# from src.dataset import ModelTreesDataLoader
# from torch.utils.data import DataLoader
from src.utils import *
from models.model import KDE_cls_model
# from time import time
from omegaconf import OmegaConf
# import argparse
from packaging import version

SRC_MODEL = "./models/pretrained/model_KDE.tar"


    
def mapToKDE(sample, kde_transform):
    pointCloud = np.asarray(sample.xyz)
    sample = {'data': pointCloud, 'label': 0}
    sample = kde_transform(sample)
    return sample


def fast_inference(samples, args):
    print("Loading model...")
    conf = {
        "num_class": args['inference']['num_class'],
        "grid_dim": args['inference']['grid_size'],
    }

    # load the model
    model = KDE_cls_model(conf).to(torch.device('cuda'))
    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        checkpoint = torch.load(SRC_MODEL, weights_only=False)
    else:
        checkpoint = torch.load(SRC_MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    kde_transform = ToKDE(
        args['inference']['grid_size'], 
        args['inference']['kernel_size'], 
        args['inference']['num_repeat_kernel'],
        )

    lst_grids = []
    for sample in samples:
        sample_kde = mapToKDE(sample, kde_transform)
        print(sample_kde.keys())
        grid, _ = sample_kde['data'], sample_kde['label']
        grid = grid.cuda()
        lst_grids.append(grid)
    if len(lst_grids) == 1:
        batch = grid.reshape(1,grid.shape[0],grid.shape[1],grid.shape[2])
    else:
        batch = torch.stack(lst_grids, axis=0)
    batch = batch.cuda()
    pred = model(batch)
    return np.array(pred.detach().cpu())


if __name__ == "__main__":
    args = OmegaConf.load('./config/inference.yaml')
    src_tile = r"D:\PDM_repo\Github\PDM\results\samples_split_fail\tile_3.laz"
    src_mask = r"D:\PDM_repo\Github\PDM\results\samples_split_fail\mask3.pickle"
    src_other_mask = r"D:\PDM_repo\Github\PDM\results\samples_split_fail\other_mask3.pickle"
    tile = laspy.read(src_tile)
    with open(src_mask, 'rb') as file:
        mask = pickle.load(file)
    with open(src_other_mask, 'rb') as file:
        other_mask = pickle.load(file)
    sample = tile[mask]
    other_sample = tile[other_mask]
    preds = fast_inference([sample, other_sample], args)
    print(np.argmax(preds, axis=1))