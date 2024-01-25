#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/21 16:03
# @Author: ZhaoKe
# @File : trainer_transformer.py
# @Software: PyCharm
import json
import os
import time
from datetime import timedelta

import numpy as np
import torch.cuda
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.optim import Adam
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from ackit.models.transformer import Transformer, subsequent_mask
from ackit.utils.metrics import accuracy
from ackit.data_utils.reader import UrbansoundDataset
from ackit.utils.utils import dict_to_object, plot_confusion_matrix


class TransformerTrainer():
    def __init__(self, configs="./configs/transformer.yaml", use_gpu=True):
        self.d_model = 512
        self.d_feedforward = 2048
        self.num_head = 8
        self.N = 6
        self.dropout = 0.1
        self.class_num = 10
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.batch_size = 30
        self.configs = None

        if isinstance(configs, str):
            with open(configs, 'r') as jsf:
                cfg = yaml.load(jsf.read(), Loader=yaml.FullLoader)
                self.configs = dict_to_object(cfg)
        else:
            self.configs = dict_to_object(configs)
        self.device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
        self.label2meta = None
        with open(self.configs.l2m, 'r') as l2m:
            self.label2meta = json.load(l2m)
        self.class_labels = []
        for key in self.label2meta:
            self.class_labels.append(self.label2meta[key])

    def __setup_loader(self, is_train=True):
        is_feat = True
        # collate_fn = collate_fn_zero2_pad if is_feat else collate_fn_zero1_pad
        if is_train:
            print("create train loader...")
            self.train_dataset = UrbansoundDataset(root=self.configs.data_root,
                                                   file_mode="train",
                                                   is_feat=is_feat)
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                           shuffle=True,
                                           num_workers=self.configs.dataset_conf.dataLoader.num_workers)
        # 获取测试数据
        self.valid_dataset = UrbansoundDataset(root=self.configs.data_root, file_mode="valid",
                                               is_feat=is_feat)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.configs.dataset_conf.dataLoader.batch_size,
                                       shuffle=True,
                                       num_workers=self.configs.dataset_conf.dataLoader.num_workers)

        print("create finished.")
    def __setup_model(self):
        self.model = Transformer(d_model=self.d_model, d_ff=self.d_feedforward, h=self.num_head, N=self.N,
                                 cls_num=self.class_num)
        self.optimizer = NoamOpt(model_size=self.d_model, factor=1, warmup=400,
                                 optimizer=Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        self.model.to(self.device)
        self.loss = SimpleLossCompute

    def __train_epoch(self, epoch_id):
        num_steps = len(self.train_loader)
        train_bar = tqdm(self.train_loader, total=num_steps, desc=f"Epoch-{epoch_id}")
        acc_list = []
        loss_list = []
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

            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()

            acc_list.append(accuracy(output, label))
            loss_list.append(loss)
            # print(f"loss: {loss}")
            loop_num += 1
        # print("acc:", acc)
        return sum(acc_list) / loop_num, sum(loss_list) / loop_num

    def train(self):
        self.__setup_loader(is_train=True)
        self.__setup_model()

        last_epoch = -1
        best_score = 0.0

        train_acc_list = []
        valid_acc_list = []
        train_loss_list = []
        valid_loss_list = []
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epochs):
            epoch_id += 1
            start_time = time.time()

            avg_acc, avg_loss = self.__train_epoch(epoch_id=epoch_id)
            # return
            val_loss, val_acc = self.evaluate(epoch_id, save_matrix_path="runs/")
            print(f"epoch[{epoch_id}], avg_acc: ", avg_acc)
            train_loss_list.append(avg_loss)
            train_acc_list.append(avg_acc)
            valid_acc_list.append(val_acc)
            valid_loss_list.append(val_loss)

            print("time cost per epoch: ", str(timedelta(seconds=(time.time() - start_time))))
            print("seconds per epoch: ", time.time() - start_time)

            if avg_acc > best_score:
                best_score = avg_acc

        # 是一维向量，可以
        np.savetxt(f"./runs/transformer/train_loss_epoch-{self.configs.train_conf.max_epochs}.txt", train_loss_list,
                   fmt="%.8e", delimiter=',', newline='\n')
        np.savetxt(f"./runs/transformer/train_acc_epoch-{self.configs.train_conf.max_epochs}.txt", train_acc_list,
                   fmt="%.8e", delimiter=',', newline='\n')
        np.savetxt(f"./runs/transformer/valid_loss_epoch-{self.configs.train_conf.max_epochs}.txt", valid_loss_list,
                   fmt="%.8e", delimiter=',', newline='\n')
        np.savetxt(f"./runs/transformer/valid_acc_epoch-{self.configs.train_conf.max_epochs}.txt", valid_acc_list,
                   fmt="%.8e", delimiter=',', newline='\n')

    def evaluate(self, epoch_id, save_matrix_path=None):
        self.model.eval()
        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (mfcc, label, _) in enumerate(tqdm(self.valid_loader)):
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


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        # if self.opt is not None:
        #     self.opt.step()
        #     self.opt.optimizer.zero_grad()
        return loss.data.item() * norm


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
