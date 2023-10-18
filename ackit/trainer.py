#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/15 17:45
# @Author: ZhaoKe
# @File : trainer.py
# @Software: PyCharm
import json
import os
import shutil
import time
from datetime import timedelta

import numpy as np
import yaml
import torch
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from ackit.models.hst import HSTModel
from ackit.models.tdnn import TDNN
from ackit.utils.logger import setup_logger
from ackit.utils.metrics import accuracy
from ackit.utils.reader import UrbansoundDataset
from ackit.utils.utils import dict_to_object, plot_confusion_matrix

logger = setup_logger(__name__)


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
        self.optimizer = None
        self.amp_scaler = None

        self.label2meta = None
        with open(self.configs.l2m, 'r') as l2m:
            self.label2meta = json.load(l2m)
        self.class_labels = []
        for key in self.label2meta:
            self.class_labels.append(self.label2meta[key])

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
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.configs.optim_conf.lr,
                                             weight_decay=self.configs.optim_conf.weight_decay)
            print("initialize loss, amp, Optimizer SGD...")

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_acc = 0
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pth'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pth'))):
            # 自动获取最新保存的模型
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pth')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pth')), "优化方法参数文件不存在！"
            state_dict = torch.load(os.path.join(resume_model, 'model.pth'))
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pth')))
            # # 自动混合精度参数
            # if self.amp_scaler is not None and os.path.exists(os.path.join(resume_model, 'scaler.pth')):
            #     self.amp_scaler.load_state_dict(torch.load(os.path.join(resume_model, 'scaler.pth')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_acc = json_data['accuracy']
            logger.info('成功恢复模型参数和优化方法参数：{}'.format(resume_model))
            self.optimizer.step()
            # [self.scheduler.step() for _ in range(last_epoch * len(self.train_loader))]
        return last_epoch, best_acc

    def __save_checkpoint(self, save_model_path: str, params: dict, best_model: bool):
        state_dict = self.model.state_dict()
        if best_model:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'best_model')
        else:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'epoch_{}'.format(params['epoch_id']))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(state_dict, os.path.join(model_path, 'model.pth'))
        # 自动混合精度参数
        # if self.amp_scaler is not None:
        #     torch.save(self.amp_scaler.state_dict(), os.path.join(model_path, 'scaler.pth'))
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            # data = {"last_epoch": epoch_id, "accuracy": best_acc, "version": __version__}
            f.write(json.dumps(params))
        if not best_model:
            last_model_path = os.path.join(save_model_path,
                                           f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                           'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path,
                                          f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                          'epoch_{}'.format(params['epoch_id'] - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))

    def __train_epoch(self, epoch_id):
        num_steps = len(self.train_loader)
        train_bar = tqdm(self.train_loader, total=num_steps, desc=f"Epoch-{epoch_id}")
        acc = []
        loop_num = 0
        for batch_id, (feat, label, _) in enumerate(train_bar):
            # step 3
            feat = feat.to(torch.float32).to(self.device)
            label = label.to(self.device).long()
            # ilr = ilr.to(self.device)
            # feat, _ = mask_with_ratio(data, ilr)
            # feat shape: torch.Size([64, 321, 40]) false
            # print("feat shape: ", feat.shape)
            # return

            # step 4
            output = self.model(feat)
            # step 5
            loss = self.loss(output, label)
            # step 6 7 8
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            acc.append(accuracy(output, label))
            # print(f"loss: {loss}")
            loop_num += 1
        # print("acc:", acc)
        return sum(acc) / loop_num

    def train(self, save_model_path="models/", resume_model=None):
        self.__setup_dataloader(is_train=True)
        self.__setup_model(is_train=True)
        if not self.configs.train_conf["train_from_zero"]:
            self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)
        last_epoch = -1
        best_score = 0.0
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epochs):
            epoch_id += 1
            start_time = time.time()
            avg_acc = self.__train_epoch(epoch_id=epoch_id)
            print(f"epoch[{epoch_id}], avg_acc: ", avg_acc)
            # return
            avg_score = self.evaluate(epoch_id, save_matrix_path="runs/")
            print("tim cost per epoch: ", timedelta(time.time() - start_time))
            # self.model.train()
            if avg_acc > best_score:
                best_score = avg_acc
                self.__save_checkpoint(save_model_path=save_model_path,
                                       params={"epoch_id": epoch_id, "accuracy": avg_acc},
                                       best_model=True)
            self.__save_checkpoint(save_model_path=save_model_path,
                                   params={"epoch_id": epoch_id, "accuracy": avg_acc},
                                   best_model=False)

    def evaluate(self, epoch_id, save_matrix_path=None):
        self.model.eval()
        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (mfcc, label, _) in enumerate(tqdm(self.test_loader)):
                mfcc = mfcc.to(self.device).to(torch.float32)
                label = label.to(self.device).long()
                output = self.model(mfcc)
                los = self.loss(output, label)

                acc = accuracy(output, label)
                accuracies.append(acc)

                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())

                labels.extend(label.tolist())
                losses.append(los.data.cpu().numpy())
        loss = float(sum(losses) / len(losses))
        acc = float(sum(accuracies) / len(accuracies))
        # 保存混合矩阵
        if save_matrix_path is not None:
            # print(f"labels length: {label.shape}, preds length: {len(preds)}, output length:{output.shape}")
            cm = confusion_matrix(labels, preds)
            plot_confusion_matrix(cm=cm,
                                  save_path=os.path.join(save_matrix_path,
                                  f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                  f'epoch_{epoch_id}-{int(time.time())}.png'),
                                  class_labels=self.class_labels)

        self.model.train()
        return loss, acc

    def test(self, resume_model):
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pth')
            assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')
        self.model.eval()
        pass
