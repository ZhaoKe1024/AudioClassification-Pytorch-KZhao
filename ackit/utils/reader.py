#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/16 8:40
# @Author: ZhaoKe
# @File : reader.py
# @Software: PyCharm
import os
import random

import librosa
import numpy as np
import yaml
from torch.utils.data import Dataset

from ackit.utils.audio import AudioSegment


class UrbansoundDataset(Dataset):
    def __init__(self, root, file_list=None):
        self.root = root
        self.file_list_path = file_list
        self.file_list = []
        self.label_list = []
        self.file_list_init()

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
        audio_path, label = self.file_list[item], self.label_list[item]
        audio_segment = AudioSegment.from_file(os.path.join(self.root, "audio", audio_path).replace('\\', '/'))
        sample, sr = audio_segment._samples, audio_segment._sr
        # print(sample.shape, sr)
        mfcc = np.mean(librosa.feature.mfcc(y=sample.mean(axis=1), sr=sr, n_mfcc=40), axis=1)
        # print(mfcc.shape)
        return np.array(mfcc, dtype=np.float32), np.array(label, dtype=np.int64)

    def __len__(self):
        return len(self.label_list)


def dataset_test():
    configs = "../../configs/hst.yaml"
    if isinstance(configs, str):
        with open(configs, 'r', encoding='utf-8') as f:
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)

    ds = UrbansoundDataset(configs['data_root'], "../../datasets/train_list.txt")
    sample, sr = ds[1]
    print(sample.shape)


if __name__ == '__main__':
    dataset_test()
