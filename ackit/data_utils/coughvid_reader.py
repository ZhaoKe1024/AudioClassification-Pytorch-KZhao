# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-24 18:17
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import librosa
import time
import torch
from torch.utils.data import Dataset, DataLoader
from ackit.data_utils.audio import AudioSegment, vad
from ackit.data_utils.audio import wav_slice_padding
from ackit.data_utils.featurizer import Wave2Mel


def ext_list():
    """{'json': 34434, 'webm': 29348, 'wav': 3309, 'ogg': 1777, 'csv': 1}"""
    root = "D:/DATAS/Medical/COUGHVID-public_dataset_v3/coughvid_20211012"
    ext_dict = dict()
    for item in os.listdir(root):
        ext = item.split('.')[-1]
        if ext not in ext_dict:
            ext_dict[ext] = 1
        else:
            ext_dict[ext] += 1
    print(ext_dict)


def stat_coughvid():
    sound_dir = "D:/DATAS/Medical/COUGHVID-public_dataset_v3/coughvid_20211012/"
    len_dict = dict()
    start_time = time.time()
    for idx, item in enumerate(os.listdir(sound_dir)):
        if idx % 100 == 0:
            print("cnt:", idx, "; cost time:", time.time() - start_time)
            print(len_dict)

        ext = item.split('.')[-1]
        if ext in ["csv", "json"]:
            continue
        dur = int(librosa.get_duration(filename=sound_dir + item))
        if dur not in len_dict:
            len_dict[dur] = 1
        else:
            len_dict[dur] += 1
    print(len_dict)


def read_labels_from_csv():
    """

    groupby: https://zhuanlan.zhihu.com/p/101284491
    :return:
    """
    metainfo_path = "G:/DATAS-Medical/COUGHVID-public_dataset_v3/coughvid_20211012/metadata_compiled.csv"
    column_names = ["uuid", "respiratory_condition", "fever_muscle_pain", "status", "status_SSL", ]
    pd_metainfo = pd.read_csv(metainfo_path, delimiter=',', header=0, index_col=0)
    pd_status = pd_metainfo[["uuid", "status"]]
    print("info of status:", len(pd_status))
    pd_status = pd_status.dropna()
    print("info of status after dropna:", len(pd_status))
    pd_status_stat = list(pd_status.groupby("status"))
    for stat_item in pd_status_stat:
        print(stat_item[0], ': ', len(stat_item[1]))
    return pd_status["status"].to_numpy()
    # print("info group by status:\n", list(pd_status_stat))


class CoughVID_Dataset(Dataset):
    def __init__(self, root_path="../../datasets/waveinfo_annotation.csv", configs=None):
        # self.w2m = (AudioFeaturizer(feature_method="MelSpectrogram"))
        self.w2m = Wave2Mel(sr=16000)
        # method_args={
        # "sample_frequency": 16000,
        #          "num_mel_bins": 80}
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.configs = configs
        self.path_list = []
        self.label_list = []
        with open(root_path, 'r') as fin:
            fin.readline()
            line = fin.readline()
            while line:
                parts = line.split(',')
                self.path_list.append(parts[1])
                self.label_list.append(np.array(parts[2], dtype=np.int64))
                line = fin.readline()
        self.wav_list = []
        self.spec_list = []
        for item in tqdm(self.path_list, desc="Loading"):
            self.append_wav(item)
            break

    def __getitem__(self, ind):
        return self.spec_list[ind], self.label_list[ind]

    def __len__(self):
        return len(self.path_list)

    def append_wav(self, file_path):
        audioseg = AudioSegment.from_file(file_path)
        audioseg.vad()
        audioseg.resample(target_sample_rate=16000)
        audioseg.crop(duration=3.0, mode="train")
        # self.wav_list.append(torch.tensor(audioseg.samples, device=self.device).to(torch.float32))
        # y = wav_slice_padding(x_wav, save_len=self.configs["feature"]["wav_length"])
        print(audioseg.samples.shape)

        # expected scalar type Double but found Float
        # 解决办法：.to(torch.float32)
        x_mel = self.w2m(torch.from_numpy(audioseg.samples).to(torch.float32))
        print(x_mel.shape)
        self.spec_list.append(torch.tensor(x_mel, device=self.device))
        return torch.tensor(x_mel, device=self.device).transpose(0, 1).to(torch.float32)


if __name__ == '__main__':
    # # ext_list()
    # # stat_coughvid()
    # label_list = read_labels_from_csv()
    # print(label_list.shape)
    cough_dataset = CoughVID_Dataset()
    print(len(cough_dataset))
    # print(cough_dataset.__getitem__(0))
    # print(cough_dataset.__getitem__(15084))
