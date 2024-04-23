#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/23 16:43
# @Author: ZhaoKe
# @File : coughcls_tdnn.py
# @Software: PyCharm
import os
import yaml
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from ackit.trainer_setting import get_model
from ackit.utils.utils import load_ckpt, setup_seed
from ackit.data_utils.soundreader2020 import get_former_loader


class TrainerEncoder(object):
    def __init__(self, configs="../configs/tdnn.yaml", istrain=True, isdemo=True):
        self.configs = None
        with open(configs) as stream:
            self.configs = yaml.safe_load(stream)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.num_epoch = self.configs["fit"]["epochs"]
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.demo_test = not istrain
        self.run_save_dir = self.configs[
                                "run_save_dir"] + '/'
        if istrain:
            self.run_save_dir += self.timestr + f'_ptvae_lr-3/'
            if not isdemo:
                os.makedirs(self.run_save_dir, exist_ok=True)

        with open("../datasets/d2020_metadata2label.json", 'r', encoding='utf_8') as fp:
            self.meta2label = json.load(fp)
        self.pretrain_model = None
        self.model = None
        self.trainloader = None