#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/15 17:43
# @Author: ZhaoKe
# @File : utils.py
# @Software: PyCharm
import librosa
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt


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
