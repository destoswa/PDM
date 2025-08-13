<img width="810" height="361" alt="first_image_final_pres_alpha_smooth" src="https://github.com/user-attachments/assets/ce758614-4bb1-4b9a-ab02-5cbc0dc334ff" />

# Self-Supervised Learning with Human Feedback for tree segmentation based on LIDAR data

### introduction
This repo contain the code of the pipeline for finetuning a segmentation model (SegmentAnyTree) into any new dataset using Weakly-Supervised Learning from Human Feedback.

Following is the abstract of the corresponding paper:

Tree instance segmentation in 3D point cloud data is of utmost importance for forest monitoring, but remains challenging due
to variety in the data caused by factors such as sensor resolution, vegetation state during acquisition, terrain characteristics, etc.
Moreover, obtaining a sufficient amount of precisely labeled data to train fully supervised instance segmentation methods is ex-
pensive. To address these challenges, this work proposes a weakly supervised approach where labels of an initial segmentation
result obtained either by a non-finetuned model or a closed form algorithm are provided as a quality rating by a human operator.
These labels are used to train a rating model, whose task is to classify a segmentation output into the same classes as specified by
the human operator. The finetuned model produces an increase of 34% in the number of correctly identified trees, coupled with a
drastic reduction in the proportion of garbage predictions, albeit challenges remain in precisely segmenting trees in areas with very
dense forest canopies or sparsely forested regions characterized by trees under two meters in height where terrain features (rocks,
shrubs, etc.) can be confused as trees.

### pretrained models
Before installing anything, the two pretrained models need to be downloaded. They can be both downloaded from the assets and need to be placed at the following locations:
- Classifier: `pdm/models/KDE_classifier/models/pretrained/model_KDE.tar`
- Segmenter: `pdm/models/SegmentAnyTree/model_file/PointGroup-PAPER.pt`

### how to install
This project works through _Docker_. So, in order to set it up, you need to follow these steps:
1) install _Docker engine_ [here](https://docs.docker.com/engine/install/)
2) in a terminal, go to the root of the project, where the _Dockerfile_ is.
3) build the image using the following command (the dot ('.') at the end is important):
```
docker build -t pipeline .
```
5) once the image is built, you can create the container with the following command:
```
docker run --gpus all --shm-size=8g -it -v <full path to the root of the project>:/home/pdm pipeline
```
6) if you want to remove the container after each usage, you can add the flag `--rm` to the previous line. Otherwise, you can just run the command `docker start -i pipeline` each time you want to reopen the container.

**!!! important: The segmenter used in this project uses a version of CUDA that is incompatible with the NVIDIA 40\*\* series. !!!**

### how to use
Each of the following process are started from the docker container.

#### pre-processing (using the _Tiles Loader_)
The _Tiles Loader_ was designed to be used to prepare the dataset for the pipeline.

It works through the call to the batch file _run_TilesLoader.sh_ with the following command:
```
bash run_TilesLoader.sh <mode> <verbose>
```
- **_verbose_** is equal to False by default so you only need to precise `True` if you want to use it.
- **_mode_** can be set between the following:
    - "tilling": to start tilling on the dataset
    - "trimming": to go through the tiles and remove the ones on-which the segmenter fails (usually, the ones on which it can not find any tree)
    - "classification": Classify segmented tiles into predefined categories (garbage, multi, single) using an external classification model, and save per-tile statistics.
    - "preprocess": apply the preprocessing set in the parameters. The possibilities are _remove_hanging_points_, _flattening_ and _remove_duplicates_.
    - "tile_and_trim": do tilling and then trimming.
    - "trimand_class": do trimming and the classification.
    - "full": do tilling, trimming, preprocessing and classification.

#### training of the pipeline
In order to train the pipeline, simply run the `./train.py` script.

You will find the different configuration parameters in the folder `./config`:
- _**classifier.yaml**_: If using the default classifier model, no need to change anything.
- _**segmenter.yaml**_: You can set here the training batch, the number of epochs per loop and number of samples per epoch. The other parameters should not be changed if using standard setup
- _**dataset.yaml**_: Set here the dataset to be used. It should be the relative path from  the root of the project to the folder containing all the tiles. Eventhough, the file_format can be precised, I don't recommend using something else than _.laz_ files.
- _**pipeline.yaml**_: In this file, are stored the config for the number of loops per tiles, the fraction of the different sets, the different processes, the location of the project if continuing from an existing one and the inference configuration.

The segmenter is not very stable so it can crash. If this happens, simply remove in the results folder, the loop(s) that was/were incomplete, set the parameter `pipeline.preload.do_continuer_from_existing` to True, while precising the path to the result folder and run again the script `train.py`

#### inference with the pipeline
Similarly to the training, to do inference with a trained pipeline, run the `./inference.py` script.

The cooresponding configuration parameters to drive the inference are located in the file `./config/pipeline.yaml` in the `inference` section. 

The two flags `is_gt` and `with_groups` are to be set if the inference is done on tiles that have ground truth and if they correspond to a grouping with a corresponding csv file assigning each tile to a group (called cluster).

The grouping is done in the jupyter notebook `./notebooks/cluster_tiles.ipynb` and the ground truth is done in the notebook `notebooks/ground_truth_generation.ipynb` with the assistance of the software _CloudCompare_ and the tool _PointCLoudCleaner_

#### additional note
It worth noting that intermediatary results are stored at different locations:
- The different versions of the pseudo-labels are stored in the results
- The different versions of the predictions and corrections are stored in the dataset (and reset from one training/inference to the other)
