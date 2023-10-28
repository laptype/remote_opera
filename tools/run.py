import json
import os
import sys

import fileinput
from mmcv import Config

# test

python_path = "/home/wangpengcheng/anaconda3/envs/opera/bin/python3"
test_py_path = "/home/wangpengcheng/tmp/remote_opera/tools/test.py"

def update_config_py(config: dict):
    file_path = config['config_path']

    # 定义要修改的变量名和新值
    for k, v in config.items():

        variable_name = k
        new_value = v
        if isinstance(new_value, str):
            new_value = f'"{new_value}"'

        # 逐行读取文件并更新变量的值
        for line in fileinput.input(file_path, inplace=True):
            if line.startswith(variable_name):
                line = f'{variable_name} = {new_value}\n'
            print(line, end='')

def read_config_py(config_path):
    cfg = Config.fromfile(config_path)
    return cfg



config_list = [
    {"config_path": "configs/wifi/four_cards_cam_trans.py", "config_tag": "1028_ep-10", "max_epochs": 10},
    {"config_path": "configs/wifi/four_cards_cam_trans.py", "config_tag": "1028_ep-15", "max_epochs": 15},
]


for config in config_list:

    config_path = config['config_path']

    update_config_py(config)

    """
        TRAIN
    """
    # os.system(f"bash tools/dist_train.sh {config_path}")

    cfg = read_config_py(config_path)
    pkl_path = os.path.join(cfg.work_dir, f"epoch_{config['max_epochs']}.pth")

    """
        TEST
    """
    os.system(f"{python_path} {test_py_path} {config_path} {pkl_path} --work-dir xxx")
