#!/usr/bin/env bash

python="/home/wangpengcheng/anaconda3/envs/opera/bin/python3"

CONFIG=$1
PORT=${PORT:-10036}

CUDA_VISIBLE_DEVICES=0,1 ${python} -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port='29501' \
        --use_env \
        /home/wangpengcheng/tmp/remote_opera/tools/train.py "$CONFIG" --no-validate --launcher pytorch
