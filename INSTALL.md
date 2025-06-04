# Install

`docker build -t difflocks-cuda .`

```
docker run --gpus all -it \        
  --runtime=nvidia \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$(pwd)/:/app/projects" \
  -v "$(pwd)/:/app/data" \
  difflocks-cuda
``` 