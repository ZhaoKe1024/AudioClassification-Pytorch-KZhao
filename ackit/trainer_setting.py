#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/16 19:19
# @Author: ZhaoKe
# @File : trainer_setting.py
# @Software: PyCharm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from ackit.models.autoencoder import ConvEncoder
from ackit.models.tdnn import TDNN
from ackit.modules.scheduler import WarmupCosineSchedulerLR
from ackit.utils.utils import weight_init


def get_model(use_model, configs, istrain=True):
    model = None
    if use_model == "conv_encoder_decoder":
        model = ConvEncoder(input_channel=1, input_length=configs["model"]["input_length"],
                            input_dim=configs["feature"]["n_mels"],
                            class_num=configs["model"]["mtid_class_num"],
                            class_num1=configs["model"]["type_class_num"])
    elif use_model == "tdnn":
        model = TDNN(num_class=configs["mtid_class_num"], input_size=configs["model"]["input_length"],
                     channels=configs["model"]["input_dim"])
        # print(model)
    else:
        raise ValueError("this model is not found!!")
    if istrain:
        model.apply(weight_init)
        # amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
    return model


def get_optimizer(use_optim, model, len_loader, configs):
    if use_optim == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=configs.optimizer_conf.learning_rate,
                                     weight_decay=configs.optimizer_conf.weight_decay)
    elif use_optim == 'AdamW':
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=configs.optimizer_conf.learning_rate,
                                      weight_decay=configs.optimizer_conf.weight_decay)
    elif use_optim == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    momentum=configs.optimizer_conf.get('momentum', 0.9),
                                    lr=configs.optimizer_conf.learning_rate,
                                    weight_decay=configs.optimizer_conf.weight_decay)
    else:
        raise Exception(f'不支持优化方法：{use_optim}')
    # 学习率衰减函数
    scheduler_args = configs.optimizer_conf.get('scheduler_args', {}) \
        if configs.optimizer_conf.get('scheduler_args', {}) is not None else {}
    if configs.optimizer_conf.scheduler == 'CosineAnnealingLR':
        max_step = int(configs.train_conf.max_epoch * 1.2) * len_loader
        scheduler = CosineAnnealingLR(optimizer=optimizer,
                                      T_max=max_step,
                                      **scheduler_args)
    elif configs.optimizer_conf.scheduler == 'WarmupCosineSchedulerLR':
        scheduler = WarmupCosineSchedulerLR(optimizer=optimizer,
                                            fix_epoch=configs.train_conf.max_epoch,
                                            step_per_epoch=len_loader,
                                            **scheduler_args)
    else:
        raise Exception(f'不支持学习率衰减函数：{configs.optimizer_conf.scheduler}')
    return optimizer, scheduler
