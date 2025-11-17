# Silent Vub

## Dataloader
Template_dir has the fotd and gsplat images. Json dir has json containing Mouth open ratio which the thresh argument decides.
```
python dataloader_gsplat.py --root_dir /app/data/vfhq/ --json_dir json_dir/ --mask_path template/mask.png --template_dir template/ --thresh 0.5
```

```
python difflocks/silent_dub/train_uv_diffusion.py \
    --root_dir /path/to/root \
    --json_dir /path/to/json \
    --mask_path /path/to/mask.png \
    --config ./configs/config_uv_conditional.json \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --mixed-precision bf16 \
    --use-tensorboard \
    --save-checkpoints \
    --save-every 10000 \
    --compile \
    --num-workers 4 \
    --seed 0 \
    --open-ratio-threshold 0.1 \
    --name uv_diffusion
```

```
accelerate launch difflocks/silent_dub/train_uv_diffusion.py \
    --root_dir <ROOT_DIR> \
    --json_dir <JSON_DIR> \
    --mask_path <MASK_PATH> \
    --config ./configs/config_uv_conditional.json \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --mixed-precision bf16 \
    --use-tensorboard \
    --save-checkpoints \
    --save-every 10000 \
    --compile \
    --name uv_diffusion
```
