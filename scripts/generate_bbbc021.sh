#!/bin/bash
# Generate a small batch of synthetic BBBC021 images on a single GPU.
# Edit configs/bbbc021_all.yaml first so the three data paths are correct,
# and set RESUME to the checkpoint downloaded by download_assets.py.
set -e

RESUME=${1:-assets/checkpoints/bbbc021/checkpoint.pth}
OUT=${2:-outputs/my_first_run}

python generate.py \
    --dataset=bbbc021 \
    --config=bbbc021_all \
    --resume="$RESUME" \
    --output_dir="$OUT" \
    --batch_size=16 \
    --fid_samples=256 \
    --use_ema \
    --edm_schedule \
    --skewed_timesteps \
    --ode_method heun2 \
    --ode_options '{"nfe": 50}' \
    --use_initial=2 \
    --noise_level=1.0 \
    --cfg_scale=0.0 \
    --compute_fid \
    --save_fid_samples
