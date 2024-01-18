#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/16 15:08
# @Author: ZhaoKe
# @File : metrics.py
# @Software: PyCharm
# 计算准确率
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
from sklearn import metrics


def accuracy(output, label):
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    label = label.data.cpu().numpy()
    acc = np.mean((output == label).astype(int))
    return acc


def get_heat_map(pred_matrix, label_vec, savepath):
    max_arg = list(pred_matrix.argmax(axis=1))
    conf_mat = metrics.confusion_matrix(max_arg, label_vec)
    df_cm = pd.DataFrame(conf_mat, index=range(conf_mat.shape[0]), columns=range(conf_mat.shape[0]))
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.xlabel("predict label")
    plt.ylabel("true label")
    plt.savefig(savepath)
    plt.close()
    # plt.show()
