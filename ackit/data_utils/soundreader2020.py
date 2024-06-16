#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/16 19:03
# @Author: ZhaoKe
# @File : sound_reader.py
# @Software: PyCharm
import random
import librosa
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from ackit.data_utils.audio import wav_slice_padding
from ackit.data_utils.featurizer import Wave2Mel


def get_former_loader(istrain=True, istest=False, configs=None, meta2label=None, isdemo=False, exclude=None,
                      iswave=False):
    # generate dataset
    # generate dataset
    print("============== DATASET_GENERATOR ==============")
    ma_id_map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
    train_loader, test_loader = None, None
    if istrain:
        print("---------------train dataset-------------")
        file_paths = []
        mtid_list = []
        mtype_list = []
        with open("../datasets/d2020_trainlist.txt", 'r') as fin:
            train_path_list = fin.readlines()
            if isdemo:
                train_path_list = random.choices(train_path_list, k=2048)
            for item in train_path_list:
                parts = item.strip().split('\t')
                machine_type_id = int(parts[1])
                if exclude:
                    if machine_type_id in exclude:
                        continue
                file_paths.append(parts[0])
                mtype_list.append(machine_type_id)
                machine_id_id = parts[2]
                meta = ma_id_map[machine_type_id] + '-id_' + machine_id_id
                mtid_list.append(meta2label[meta])
        if iswave:
            train_dataset = WaveReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                                       y_true_list=None,
                                       configs=configs,
                                       istrain=True, istest=istest)
        else:
            train_dataset = FormerReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                                         y_true_list=None,
                                         configs=configs,
                                         istrain=True)
        train_loader = DataLoader(train_dataset, batch_size=configs["fit"]["batch_size"], shuffle=True)
    if istest:
        print("---------------test dataset-------------")
        file_paths = []
        mtid_list = []
        mtype_list = []
        y_true_list = []
        with open("./datasets/test_list.txt", 'r') as fin:
            test_path_list = fin.readlines()
            if isdemo:
                test_path_list = random.choices(test_path_list, k=20)
            for item in test_path_list:
                parts = item.strip().split('\t')
                machine_type_id = int(parts[1])
                if exclude:
                    if machine_type_id in exclude:
                        continue
                mtype_list.append(machine_type_id)
                file_paths.append(parts[0])
                machine_id_id = parts[2]
                y_true_list.append(int(parts[3]))
                meta = ma_id_map[machine_type_id] + '-id_' + machine_id_id
                mtid_list.append(meta2label[meta])
        if iswave:
            test_dataset = WaveReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                                      y_true_list=y_true_list,
                                      configs=configs, istrain=False, istest=istest)

        else:
            test_dataset = FormerReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                                        y_true_list=None,
                                        configs=configs,
                                        istrain=True)
        test_loader = DataLoader(test_dataset, batch_size=configs["fit"]["batch_size"], shuffle=True)
    return train_loader, test_loader


class WaveReader(Dataset):
    def __init__(self, file_paths, mtype_list, mtid_list, y_true_list, configs, istrain=True, istest=False):
        self.files = file_paths
        self.mtids = mtid_list
        self.mtype_list = mtype_list
        self.y_true = y_true_list
        self.configs = configs
        self.wav_list = []
        for fi in tqdm(file_paths, desc=f"build Set..."):
            self.wav_list.append(self.load_wav_2mel(fi))
        self.istrain = istrain
        self.istest = istest

    def __getitem__(self, ind):
        if not self.istest:
            if self.istrain:
                return self.wav_list[ind], self.mtype_list[ind], self.mtids[ind]
            else:
                return self.wav_list[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
        else:
            return self.wav_list[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]

    def __len__(self):
        return len(self.files)

    def load_wav_2mel(self, wav_path):
        # print(wav_path)
        y, sr = librosa.core.load(wav_path, sr=16000)
        y = wav_slice_padding(y, save_len=self.configs["feature"]["wav_length"])
        return y


class FormerReader(Dataset):
    def __init__(self, file_paths, mtype_list, mtid_list, y_true_list, configs, istrain=True, istest=False):
        self.files = file_paths
        self.mtids = mtid_list
        self.mtype_list = mtype_list
        self.y_true = y_true_list
        self.configs = configs
        self.w2m = Wave2Mel(16000)
        self.device = torch.device("cuda")
        self.mel_specs = []
        for fi in tqdm(file_paths, desc=f"build Set..."):
            self.mel_specs.append(self.load_wav_2mel(fi))
        self.istrain = istrain
        self.istest = istest

    def __getitem__(self, ind):
        if not self.istest:
            if self.istrain:
                return self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind]
            else:
                return self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
        else:
            return self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]

    def __len__(self):
        return len(self.files)

    def load_wav_2mel(self, wav_path):
        # print(wav_path)
        y, sr = librosa.core.load(wav_path, sr=16000)
        y = wav_slice_padding(y, save_len=self.configs["feature"]["wav_length"])
        x_mel = self.w2m(torch.from_numpy(y.T))
        return torch.tensor(x_mel, device=self.device).transpose(0, 1).to(torch.float32)


def num_bound():
    import os

    """
    [(tensor(-66.9853), tensor(26.1165)), 
    (tensor(-79.7177), tensor(26.1165)), 
    (tensor(-79.7177), tensor(26.3636)), 
    (tensor(-100.), tensor(27.5290)), 
    (tensor(-100.), tensor(39.9690)), 
    (tensor(-100.), tensor(39.9690))]
    """
    # name = "gearbox"
    mode = "train"
    mtypes = ["fan", "pump", "slider", "ToyCar", "ToyConveyor", "valve"]
    # item_name = "section_00_source_train_normal_0008_noAttribute" + ".wav"
    res = []
    wav2mel = Wave2Mel(sr=16000)
    val_min, val_max = 256., -256.
    for mt in mtypes:
        root_path = f"F:/DATAS/DCASE2020Task2ASD/dataset/dev_data_{mt}/{mt}/{mode}/"
        for i, path_item in enumerate(os.listdir(root_path)):
            x, _ = librosa.core.load(root_path + path_item, sr=16000, mono=True)
            x_wav = torch.from_numpy(x)
            x_mel = wav2mel(x_wav)
            tmp_min, tmp_max = x_mel.min(), x_mel.max()
            val_min = val_min if val_min < tmp_min else tmp_min
            val_max = val_max if val_max > tmp_max else tmp_max
            if i % 500 == 0:
                print("min max:", val_min, val_max)
        res.append((val_min, val_max))
    print("min max:", val_min, val_max)
    print(res)


if __name__ == '__main__':
    # import math
    # print(math.floor(-58.0754), math.ceil(33.5727))
    num_bound()
