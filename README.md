<img width="810" height="361" alt="first_image_final_pres_alpha_smooth" src="https://github.com/user-attachments/assets/ce758614-4bb1-4b9a-ab02-5cbc0dc334ff" />

# Self-Supervised Learning with Human Feedback for tree segmentation based on LIDAR data

### introduction

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


#### inference with the pipeline

  
