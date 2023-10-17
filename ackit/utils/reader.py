#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/16 8:40
# @Author: ZhaoKe
# @File : reader.py
# @Software: PyCharm
import os

import librosa
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from ackit.utils.audio import AudioSegment


class UrbansoundDataset(Dataset):
    def __init__(self, root, file_list=None, is_feat=False):
        self.root = root
        self.file_list_path = file_list
        self.file_list = []
        self.label_list = []
        self.file_list_init()
        self.is_feat = is_feat

    def file_list_init(self):
        with open(self.file_list_path, 'r') as tr_list:
            line = tr_list.readline()
            while line:
                parts = line.split('\t')
                self.file_list.append(parts[0])
                self.label_list.append(parts[1])
                line = tr_list.readline()

        # print(os.path.join(self.root, "/metadata/UrbanSound8K.csv").replace('\\', '/'))
        # with open(self.root + "/metadata/UrbanSound8K.csv", 'r') as csvfile:
        #     csvfile.readline()
        #     line = csvfile.readline()
        #     while line:
        #         # print(line)
        #         parts = line.split(',')
        #         file_path = os.path.join(self.root, "audio", f"fold{parts[5]}", parts[0]).replace('\\', '/')
        #         if file_path[-3:] == "wav":
        #             self.file_list.append(file_path)
        #             self.label_list.append(int(parts[6]))
        #         line = csvfile.readline()

    def __getitem__(self, item):
        """ 由于音频数据长度不一，MFCC特征的第2维度(index=1)长度也不一样，需要collate_fn处理"""
        audio_path, label = self.file_list[item], self.label_list[item]
        audio_segment = AudioSegment.from_file(
            file=os.path.join(self.root, "audio", audio_path).replace('\\', '/'))
        if audio_segment.sr != 16000:
            audio_segment.resample(target_sr=16000)
        sample, sr = audio_segment.samples.T, audio_segment.sr
        if self.is_feat:
            # sample shape: (x, 2)
            if sample.ndim >= 2:
                mfcc = librosa.feature.mfcc(y=sample.mean(axis=1), sr=sr, n_mfcc=40)
            else:
                # print(sample.shape, sr)
                mfcc = librosa.feature.mfcc(y=sample, sr=sr, n_mfcc=40)
            # print(mfcc.shape)
            # mfcc = np.mean(mfcc, axis=1)
            # print(mfcc.shape)
            return np.array(mfcc, dtype=np.float32), np.array(label, dtype=np.int64)
        else:
            if sample.ndim >= 2:
                sample = sample.mean(axis=1)
            return np.array(sample, dtype=np.float32), np.array(label, dtype=np.int64)

    def __len__(self):
        return len(self.label_list)


def collate_fn_zero1_pad(batch):
    batch_size = len(batch)
    # print("length of batch:", batch_size)  # batch_size, (item0, item1, ...)
    # print("item kinds of batch:", len(batch[0]))  # 2, (data, label)
    # print("shape of data: ", batch[0][0].shape)  # (40, x), x is to be aligned
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    max_audio_length = batch[0][0].shape[0]  # length to pad to
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
    labels = []
    input_lens_ratio = []
    for i in range(batch_size):
        sample = batch[i]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[0]
        inx_start = (max_audio_length - seq_length) // 2
        # 将数据插入都0张量中，实现了padding
        inputs[i, inx_start:inx_start + seq_length] = tensor
        input_lens_ratio.append(seq_length / max_audio_length)
    labels = np.array(labels, dtype='int64')
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens_ratio)


# 对一个batch的数据处理
# 填充到最大零矩阵的中央靠左位置
def collate_fn_zero2_pad(batch):
    batch_size = len(batch)
    # print("length of batch:", bs)  # batch_size, (item0, item1, ...)
    # print("item kinds of batch:", len(batch[0]))  # 2, (data, label)
    # print("shape of data: ", batch[0][0].shape)  # (40, x), x is to be aligned
    # 找出音频长度最长的
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    shape0, max_audio_length = batch[0][0].shape  # length to pad to
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, shape0, max_audio_length), dtype='float32')
    labels = []
    for i in range(batch_size):
        sample = batch[i]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[1]
        inx_start = (max_audio_length - seq_length) // 2
        # 将数据插入都0张量中，实现了padding
        inputs[i, :, inx_start:inx_start + seq_length] = tensor
    labels = np.array(labels, dtype='int64')
    return torch.tensor(inputs), torch.tensor(labels)


def dataset_test():
    configs = "../../configs/hst.yaml"
    if isinstance(configs, str):
        with open(configs, 'r', encoding='utf-8') as f:
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)

    ds = UrbansoundDataset(configs['data_root'], "../../datasets/train_list.txt", is_feat=False)
    dl = DataLoader(ds,
                    batch_size=16,
                    shuffle=True, num_workers=0, collate_fn=collate_fn_zero1_pad)
    from featurizer import AudioFeaturizer
    af = AudioFeaturizer(feature_method="MFCC")
    for idx, (data, label, dlr) in enumerate(dl):
        print(idx, '\t', data.shape, '\t', len(dlr))
        # x = torch.tensor(data)
        feat = af(data, dlr)
        print(feat[0].shape, feat[1].shape)
    # sample1, sr1 = ds[1]
    # sample2, sr2 = ds[2]
    # print(sample1.shape)
    # print(sample2.shape)


if __name__ == '__main__':
    dataset_test()
    # a = torch.randn((3, 3))
    # b = torch.zeros((3, 8))
    # b[:, 2:5] = a
    # print(b)
