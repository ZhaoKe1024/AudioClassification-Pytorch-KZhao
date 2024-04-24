# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-24 18:17
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import librosa
import time
from torch.utils.data import Dataset
from ackit.data_utils.audio import AudioSegment


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


def CoughVID_Lists(filename="../../datasets/waveinfo_annotation.csv", istrain=True, isdemo=False):
    path_list = []
    label_list = []
    with open(filename, 'r') as fin:
        fin.readline()
        line = fin.readline()
        ind = 0
        while line:
            parts = line.split(',')
            path_list.append(parts[1])
            label_list.append(np.array(parts[2], dtype=np.int64))
            line = fin.readline()
            ind += 1
            if isdemo:
                if ind > 1000:
                    return path_list, label_list
    N = len(path_list)
    tr, va = N * 0.8, N * 0.9
    train_path, train_label = path_list[0:tr], label_list[0:tr]
    valid_path, valid_label = path_list[tr:va], label_list[tr:va]
    if istrain:
        return train_path, train_label, valid_path, valid_label
    else:
        return path_list[va:], label_list[va:]


class CoughVID_Dataset(Dataset):
    def __init__(self, path_list, label_list):
        self.path_list = path_list
        self.label_list = label_list
        self.wav_list = []
        for item in tqdm(path_list, desc="Loading"):
            self.append_wav(item)

    def __getitem__(self, ind):
        return self.wav_list[ind], self.label_list[ind]

    def __len__(self):
        return len(self.path_list)

    def append_wav(self, file_path):
        audioseg = AudioSegment.from_file(file_path)
        audioseg.vad()
        audioseg.resample(target_sample_rate=16000)
        audioseg.crop(duration=3.0, mode="train")
        self.wav_list.append(audioseg.samples)


if __name__ == '__main__':
    # # ext_list()
    # # stat_coughvid()
    # label_list = read_labels_from_csv()
    # print(label_list.shape)
    from torch.utils.data import DataLoader
    from ackit.data_utils.collate_fn import collate_fn
    from ackit.data_utils.featurizer import Wave2Mel

    cough_dataset = CoughVID_Dataset()
    w2m = Wave2Mel(sr=16000, n_mels=80)
    train_loader = DataLoader(cough_dataset, batch_size=32, shuffle=False,
                              collate_fn=collate_fn)
    for i, (x_wav, y_label, max_len_rate) in enumerate(train_loader):
        print(x_wav.shape)
        print(y_label)
        print(max_len_rate)
        x_mel = w2m(x_wav)
        print(x_mel[0])
        break
    # print(cough_dataset.__getitem__(15084))
