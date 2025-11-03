# Install

`docker build -t difflocks-cuda .`

docker run -p 6006:6006 --user root --gpus all -it --runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=all  --shm-size=16g -v "$(pwd):/app/projects" -v "/devesh2/finetuned_fotd_silent_vub_2/repo:/app/data" difflocks-cuda` 

``` docker run -p 6006:6006 --gpus all -it --runtime=nvidia --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=all --shm-size=16g -v "$(pwd):/app/data" difflocks-github ```

```
tensorboard --logdir tensorboard_logs/hair_exp_8/ --host 0.0.0.0 --port 6006
```