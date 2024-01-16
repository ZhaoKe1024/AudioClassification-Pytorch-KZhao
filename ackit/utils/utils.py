#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/15 17:43
# @Author: ZhaoKe
# @File : utils.py
# @Software: PyCharm
import os

import librosa
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        # if any(m.bias):
        # torch.nn.init.constant_(m.bias, 0.)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1.)
        # torch.nn.init.constant_(m.bias, 0.)


def load_yaml(file_path='./config.yaml'):
    with open(file_path, encoding='utf_8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def set_type(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        return value


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst


def mask_with_start_index(feature, start_index):
    """从AudioSegment中提取音频特征

    :param feature: Audio segment to extract features from. (40, 174)
    :type feature: AudioSegment
    :param start_index: input length ratio
    :type start_index: tensor
    :return: Spectrogram audio feature in 2darray.
    :rtype: ndarray
    """
    feature = feature.transpose(2, 1)  # (174, 40)  -> (40, 174)
    # 归一化
    feature = feature - feature.mean(1, keepdim=True)  # (128, 40)  -> (128, 174)
    # print(idxs_start < idxs < (idxs_start + mask_lens))  # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
    mask = np.zeros_like(feature)
    mask = mask.unsqueeze(-1)
    # 对特征进行掩码操作
    feature_masked = torch.where(mask, feature, torch.zeros_like(feature))
    return feature_masked


def plot_confusion_matrix(cm, save_path, class_labels, title='Confusion Matrix', show=False):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(class_labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val] / (np.sum(cm[:, x_val]) + 1e-6)
        # 忽略值太小的
        if c < 1e-4: continue
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    m = np.max(cm)
    plt.imshow(cm / m, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(class_labels)))
    plt.xticks(xlocations, class_labels, rotation=90)
    plt.yticks(xlocations, class_labels)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(class_labels))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    # 保存图片
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png')
    if show:
        # 显示图片
        plt.show()



def demo_plot_spec():
    file_path = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold1/7061-6-0-0.wav"
    sample, sr = librosa.load(file_path)
    x_mfcc = librosa.feature.mfcc(y=sample, sr=sr, n_mfcc=40)
    print(x_mfcc.shape)
    plt.figure(0)
    plt.imshow(np.array(x_mfcc, dtype="uint8"))
    plt.show()


if __name__ == '__main__':
    demo_plot_spec()
