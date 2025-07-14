import os
import pandas as pd
import shutil

from tqdm import tqdm
from src.dataset import ModelTreesDataLoader
from torch.utils.data import DataLoader
from src.utils import *
from models.model import KDE_cls_model
from time import time
from omegaconf import OmegaConf
import argparse
from packaging import version


# ===================================================
# ================= HYPERPARAMETERS =================
# ===================================================
# preprocessing
# do_preprocess = True
# do_update_caching = True

# inference
# batch_size = 8
# num_workers = 8
# num_class = 3
# grid_size = 64
# kernel_size = 1
# num_repeat_kernel = 2
SRC_INF_ROOT = "./inference/"
SRC_INF_DATA = SRC_INF_ROOT + "data/"
SRC_MODEL = "./models/pretrained/model_KDE.tar"
INFERENCE_FILE = "modeltrees_inference.csv"
with open('./inference/modeltrees_shape_names.txt', 'r') as f:
    SAMPLE_LABELS = f.read().splitlines()

# ===================================================
# ===================================================

# store relation between number and class label
dict_labels = {}
for idx, cls in enumerate(SAMPLE_LABELS):
    dict_labels[idx] = cls


def inference(args):
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

    # create the folders for results
    if not os.path.isdir(os.path.normpath(args['inference']['src_root_data']) + '/results'):
        os.mkdir(os.path.normpath(args['inference']['src_root_data']) + '/results')
    for cls in SAMPLE_LABELS:
        if not os.path.isdir(os.path.normpath(args['inference']['src_root_data']) + '/results/' + cls):
            os.mkdir(os.path.normpath(args['inference']['src_root_data']) + '/results/' + cls)

    # preprocess the samples
    if args['inference']['do_preprocess']:
        lst_files_to_process = ['data/' + cls for cls in os.listdir(args['inference']['src_data']) if cls.endswith('pcd')]
        df_files_to_process = pd.DataFrame(lst_files_to_process, columns=['data'])
        df_files_to_process['label'] = 0
        df_files_to_process.to_csv(os.path.join(args['inference']['src_root_data'], INFERENCE_FILE), sep=';', index=False)

    # make the predictions
    print("making predictions...")
    kde_transform = ToKDE(
        args['inference']['grid_size'], 
        args['inference']['kernel_size'], 
        args['inference']['num_repeat_kernel'],
        )
    inferenceSet = ModelTreesDataLoader(INFERENCE_FILE, 
                                        args['inference']['src_root_data'], 
                                        split='inference', 
                                        transform=None, 
                                        do_update_caching=args['inference']['do_update_caching'],
                                        kde_transform=kde_transform,
                                        )
    inferenceDataLoader = DataLoader(inferenceSet, 
                                     batch_size=args['inference']['batch_size'], 
                                     shuffle=False, num_workers=args['inference']['num_workers'], 
                                     pin_memory=True,
                                     )
    df_predictions = pd.DataFrame(columns=["file_name", "class"])

    for _, data in tqdm(enumerate(inferenceDataLoader, 0), total=len(inferenceDataLoader), smoothing=0.9):
        # load the samples and labels on cuda
        grid, target, filenames = data['grid'], data['label'], data['filename']
        grid, target = grid.cuda(), target.cuda()

        # compute prediction
        pred = model(grid)
        pred_choice = pred.data.max(1)[1]

        for idx, pred in enumerate(pred_choice):
            fn = filenames[idx].replace('.pickle', '')
            dest = os.path.join(args['inference']['src_root_data'], 'results/', dict_labels[pred.item()] + "/" + fn.replace('data/', ""))
            shutil.copyfile(os.path.join(args['inference']['src_root_data'], fn), dest)
            df_predictions.loc[len(df_predictions)] = [fn, pred.item()]

    # save results in csv file
    df_predictions.to_csv(os.path.join(args['inference']['src_root_data'], 'results/results.csv'), sep=';', index=False)

    # Remove temp folder
    inferenceSet.clean_temp()


def main():
    args = OmegaConf.load('./config/inference.yaml')
    parser = argparse.ArgumentParser(description="Classify tree prediction based on lidar data")
    parser.add_argument('--src_root_data', type=str, default=None)
    parser.add_argument('--src_data', type=str, default=None)
    args_modif = parser.parse_args()
    if args_modif.src_root_data != None:
        args.inference.src_root_data = args_modif.src_root_data
    if args_modif.src_data != None:
        args.inference.src_data = args_modif.src_data
    inference(args)


if __name__ == "__main__":
    start = time()
    main()
    duration = time() - start
    hours = int(duration/3600)
    mins = int((duration - 3600 * hours)/60)
    secs = int((duration - 3600 * hours - 60 * mins))
    print(duration)
    print(f"Time to process inference: {hours}:{mins}:{secs}")
