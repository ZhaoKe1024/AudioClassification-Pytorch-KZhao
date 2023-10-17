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
from ackit.models.tdnn import TDNN
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
        is_feat = True
        # collate_fn = collate_fn_zero2_pad if is_feat else collate_fn_zero1_pad
        if is_train:
            self.train_dataset = UrbansoundDataset(root=self.configs.data_root,
                                                   file_list="train",
                                                   is_feat=is_feat)
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                           shuffle=True,
                                           num_workers=self.configs.dataset_conf.dataLoader.num_workers)
            print("create train valid test loader...")
        # 获取测试数据
        self.valid_dataset = UrbansoundDataset(root=self.configs.data_root, file_list="valid",
                                               is_feat=is_feat)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                       shuffle=True,
                                       num_workers=self.configs.dataset_conf.dataLoader.num_workers)

        self.test_dataset = UrbansoundDataset(root=self.configs.data_root, file_list="test",
                                              is_feat=is_feat)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                      shuffle=False,
                                      num_workers=self.configs.dataset_conf.dataLoader.num_workers)

    def __setup_model(self, is_train):
        if self.configs.use_model == "hst":
            self.model = HSTModel(self.configs)
        elif self.configs.use_model == "tdnn":
            print("initialize TDNN model...")
            self.model = TDNN(num_class=10, input_size=40)  # [16, 40, 126]
        self.model.to(self.device)

        # self.audio_featurizer = AudioFeaturizer(feature_method=self.configs.preprocess_conf["feature_method"])
        # self.audio_featurizer.to(self.device)
        # print("initialize Audio Featurizer...")

        self.loss = F.cross_entropy
        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
            self.optim = torch.optim.SGD(self.model.parameters(), self.configs.optim_conf.lr,
                                         weight_decay=self.configs.optim_conf.weight_decay)
            print("initialize loss, amp, Optimizer SGD...")

    def __save_checkpoint(self):
        pass

    def __train_epoch(self, epoch_id):
        num_steps = len(self.train_loader)
        train_bar = tqdm(self.train_loader, total=num_steps, desc=f"Epoch-{epoch_id}")
        acc = []
        loop_num = 0
        for batch_id, (feat, label, _) in enumerate(train_bar):
            # step 3
            feat = feat.to(torch.float32).to(self.device)
            label = label.to(self.device)
            # ilr = ilr.to(self.device)
            # feat, _ = mask_with_ratio(data, ilr)
            # feat shape: torch.Size([64, 321, 40]) false
            # print("feat shape: ", feat.shape)
            # return

            # step 4
            output = self.model(feat)
            # step 5
            loss = self.loss(output, label.long())
            # step 6 7 8
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            acc.append(accuracy(output, label))
            # print(f"loss: {loss}")
        # print("acc:", acc)
        return sum(acc) / loop_num

    def train(self):
        self.__setup_dataloader(is_train=True)
        self.__setup_model(is_train=True)
        last_epoch = -1
        best_score = 0.0
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epochs):
            epoch_id += 1
            start_time = time.time()
            avg_acc = self.__train_epoch(epoch_id=epoch_id)
            print(avg_acc)
            # return
            # avg_score = self.evaluate()
            print("tim cost per epoch: ", timedelta(time.time() - start_time))
            # self.model.train()
            # if avg_score > best_score:
            #     best_score = avg_score
            #     self.__save_checkpoint()

    def evaluate(self):
        return 0.0

    def test(self):
        pass
