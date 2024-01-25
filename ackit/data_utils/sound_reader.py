#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/16 19:03
# @Author: ZhaoKe
# @File : sound_reader.py
# @Software: PyCharm
import random
import numpy as np
import librosa
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from ackit.data_utils.audio import wav_slice_padding
from ackit.data_utils.featurizer import Wave2Mel


def get_former_loader(istrain=True, istest=False, configs=None, meta2label=None, isdemo=False, exclude=None):
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
        with open("./datasets/train_list.txt", 'r') as fin:
            train_path_list = fin.readlines()
            if isdemo:
                train_path_list = random.choices(train_path_list, k=2000)
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
        train_dataset = FormerReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list, y_true_list=None,
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
        test_dataset = FormerReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list, y_true_list=y_true_list,
                                    configs=configs, istrain=False, istest=istest)
        test_loader = DataLoader(test_dataset, batch_size=configs["fit"]["batch_size"], shuffle=True)
    return train_loader, test_loader


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
        self.istest=istest

    def __getitem__(self, ind):
        if not self.istest:
            if self.istrain:
                # print("***")
                return self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind]
            else:
                # print("????")
                return self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
            # if self.istrain:
            #     return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.mtype_list[ind], self.mtids[ind]
            # else:
            #     return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
        else:
            # print("!!!")
            # print(self.mel_specs[ind].shape, self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind])
            return self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
            # return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.y_true[ind], self.files[ind]

    def __len__(self):
        return len(self.files)

    def load_wav_2mel(self, wav_path):
        # print(wav_path)
        y, sr = librosa.core.load(wav_path, sr=16000)
        y = wav_slice_padding(y, save_len=self.configs["feature"]["wav_length"])
        x_mel = self.w2m(torch.from_numpy(y.T))
        return torch.tensor(x_mel, device=self.device).transpose(0, 1).to(torch.float32)
