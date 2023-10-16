#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/15 17:45
# @Author: ZhaoKe
# @File : trainer_hst.py
# @Software: PyCharm
import time
from datetime import timedelta

import yaml
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from ackit.models.hst import HSTModel
from ackit.utils.metrics import accuracy
from ackit.utils.reader import UrbansoundDataset
from ackit.utils.utils import dict_to_object


class HSTTrainer(object):
    def __init__(self, configs, use_gpu=True):
        self.configs = None
        if isinstance(configs, str):
            with open(configs, 'r') as jsf:
                cfg = yaml.load(jsf.read(), Loader=yaml.FullLoader)
                self.configs = dict_to_object(cfg)
        else:
            self.configs = dict_to_object(configs)
        self.device = torch.device("cuda:0") if use_gpu else torch.device("cpu")

        self.train_loader = None
        self.valid_loader = None

        self.model = None
        self.optim = None
        self.amp_scaler = None


    def print_configs(self):
        # for item in self.configs.keys():
        #     print(self.configs[item])
        print(self.configs)

    def __setup_dataloader(self, is_train):
        if is_train:
            self.train_dataset = UrbansoundDataset(root=self.configs.data_root, file_list=self.configs.train_list)
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                           shuffle=True, num_workers=self.configs.dataset_conf.dataLoader.num_workers)

        # 获取测试数据
        self.valid_dataset = UrbansoundDataset(root=self.configs.data_root, file_list=self.configs.valid_list)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                       shuffle=True, num_workers=self.configs.dataset_conf.dataLoader.num_workers)

        self.test_dataset = UrbansoundDataset(root=self.configs.data_root, file_list=self.configs.test_list)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                      shuffle=False, num_workers=self.configs.dataset_conf.dataLoader.num_workers)

    def __setup_model(self, is_train):
        if self.configs.use_model == "hst":
            self.model = HSTModel(self.configs)
        self.model.to(self.device)
        self.loss = F.cross_entropy
        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
            self.optim = torch.optim.SGD(self.model.parameters(), self.configs.optim_conf.lr,
                                         weight_decay=self.configs.optim_conf.weight_decay)

    def __save_checkpoint(self):
        pass

    def __train_epoch(self, epoch_id):
        num_steps = len(self.train_loader)
        train_bar = tqdm(self.train_loader, total=num_steps, desc=f"Epoch-{epoch_id}")
        for batch_id, (x_mfcc, label) in enumerate(train_bar):
            # step 3
            x_mfcc = x_mfcc.to(self.device)
            label = label.to(self.device)
            # step 4
            output = self.model(x_mfcc)
            # step 5
            loss = self.loss(output, label)
            # step 6 7 8
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            acc = accuracy(x_mfcc, label)

    def train(self):
        self.__setup_dataloader(is_train=True)
        self.__setup_model(is_train=True)
        last_epoch = -1
        best_score = 0.0
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epochs):
            epoch_id += 1
            start_time = time.time()
            self.__train_epoch(epoch_id=epoch_id)

            # avg_score = self.evaluate()
            # print(timedelta(time.time() - start_time))
            # self.model.train()
            # if avg_score > best_score:
            #     best_score = avg_score
            #     self.__save_checkpoint()

    def evaluate(self):
        return 0.0

    def test(self):
        pass
