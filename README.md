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
### how to use

#### pre-processing (using the _Tiles Loader_)

#### training of the pipeline

#### Pipeline on Ground Truth

#### evaluation (using the _Tiles Loader_)
  
