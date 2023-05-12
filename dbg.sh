#!/usr/bin/env bash
touch ~/wait1
torchrun --nproc_per_node=1 --nnodes=1 --rdzv_id=918291 --rdzv-backend=c10d --rdzv-endpoint=localhost:36593 train.py --outdir ./output_dbg/ --cfg lite --data ./data/cifar10-32x32.zip --img-resolution 32 --batch 16 --batch-gpu 4 --kimg 25000 --metrics fid50k_full
rm -rf ~/wait1
