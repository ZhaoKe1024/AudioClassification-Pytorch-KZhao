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
from torch.utils.data import Dataset

from ackit.utils.audio import AudioSegment


class UrbansoundDataset(Dataset):
    def __init__(self, root, file_mode="train", is_feat=False):
        self.root = root
        self.file_list_path = file_mode
        self.is_feat = is_feat
        self.file_list = None
        self.label_list = None
        self.pad_start = None
        self.file_list_init_02(mode=file_mode)

    def __getitem__(self, ind):
        return self.file_list[ind], self.label_list[ind], self.pad_start[ind]

    """
    version 0.2
    read the aligned MFCC file which is preprocessed already.
    # @datetime: 2023-10-20 09:54 program block accomplished
    """
    def _getitem_from_mfcc_aligned(self, idx):
        return self.file_list[idx], self.label_list[idx], self.pad_start[idx]

    # @datetime: 2023-10-20 09:54 program block accomplished
    def file_list_init_02(self, mode="train"):
        self.label_list = []
        self.pad_start = []
        self.file_list = []
        print(f"reading {mode} mfcc...")
        with open(f"./datasets/{mode}_vector.txt", 'r') as vec_file:
            line = vec_file.readline()
            while line:
                parts = line.strip().split(' ')
                # print(len(parts))  # 120003
                self.label_list.append(int(parts[0]))
                self.pad_start.append(int(parts[1]))
                vector = np.array([float(x) for x in parts[2:]])
                assert len(vector) == 40*3000, "the length is not matched"
                self.file_list.append(vector.reshape(40, 3000))
                line = vec_file.readline()
        print(f"load {mode} mfcc vector:", len(self.file_list))
        # print()

    def file_list_init_01(self):
        if self.is_feat:
            file_mfccs = np.load(f"./datasets/{self.file_list_path}_mfcc.npy")
            file_info = np.load(f"./datasets/{self.file_list_path}_info.npy ")
            # self.
            self.file_list = file_mfccs
            self.label_list = file_info[:, 0]
            self.pad_start = file_info[:, 1]
        else:
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

    """
     version 0.1
     根据音频路径列表，读取音频，转换为MFCC，然后通过collate_fn对齐MFCC
     效率很低，速度很慢，IO跟不上GPU(4090)
     reading the audio according to the audio path, then convert it to MFCC,
     and then align MFCC through collate_fn,
     but it is very low efficiency, slow speed, the speed of IO cannot keep up with GPU.
    """
    def _getitem_from_wavfile(self, item):
        """废弃了! 除非返回在这里读取音频,但是真慢"""
        """ 由于音频数据长度不一，MFCC特征的第2维度(index=1)长度也不一样，需要collate_fn处理"""
        audio_path, label = self.file_list[item], self.label_list[item]
        audio_segment = AudioSegment.from_file(
            file=os.path.join(self.root, "audio", audio_path).replace('\\', '/'))
        if audio_segment._sample_rate != 16000:
            audio_segment.resample(target_sample_rate=16000)
        sample, sr = audio_segment._samples.T, audio_segment._sample_rate
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
