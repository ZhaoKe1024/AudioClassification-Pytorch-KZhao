#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/16 15:08
# @Author: ZhaoKe
# @File : metrics.py
# @Software: PyCharm
# 计算准确率
import numpy as np
import torch


def accuracy(output, label):
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    label = label.data.cpu().numpy()
    acc = np.mean((output == label).astype(int))
    return acc