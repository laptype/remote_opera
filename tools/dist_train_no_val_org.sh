#!/usr/bin/env bash

python_path="/home/wangpengcheng/anaconda3/envs/opera/bin/python3"

CONFIG=$1
GPUS=$2
PORT=${PORT:-10036}

PYTHONPATH=python_path:$PYTHONPATH \
python -m torch.distributed.launch \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        --use_env \
        $(dirname "$0")/train.py $CONFIG --no-validate --launcher pytorch ${@:3}
