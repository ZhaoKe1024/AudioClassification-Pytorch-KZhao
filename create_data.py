#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/14 11:13
# @Author: ZhaoKe
# @File : create_data.py
# @Software: PyCharm
import os
import numpy as np
import librosa
import pandas as pd
import soundfile


def ext_list():
    root = "E:/DATAS/medicaldata/COUGHVID-public_dataset_v3/coughvid_20211012"
    ext_dict = dict()
    for item in os.listdir(root):
        ext = item.split('.')[-1]
        if ext not in ext_dict:
            ext_dict[ext] = 1
        else:
            ext_dict[ext] += 1
    print(ext_dict)


def create_file_list(metafile_path):
    root = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K"
    # train_list = []
    # valid_list = []
    # test_list = []
    train_file = open("datasets/train_list.txt", 'w')
    valid_file = open("datasets/valid_list.txt", 'w')
    test_file = open("datasets/test_list.txt", 'w')
    with open(os.path.join(root, metafile_path), 'r') as csvfile:
        csvfile.readline()
        line = csvfile.readline()
        ind = 1
        while line:
            # print(line)
            parts = line.split(',')
            file_path = f"fold{parts[5]}/" + parts[0]
            if file_path[-3:] == "wav":
                if ind % 10 == 8:
                    valid_file.write(file_path+'\t'+parts[6]+'\n')
                elif ind % 10 == 9:
                    test_file.write(file_path+'\t'+parts[6]+'\n')
                else:
                    train_file.write(file_path+'\t'+parts[6]+'\n')
            line = csvfile.readline()
            ind += 1
            # break
    test_file.close()
    valid_file.close()
    train_file.close()


def create_mfcc_npy_data(dataest_path):
    root = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K"
    with open(dataest_path, 'r') as csvfile:
        csvfile.readline()
        line = csvfile.readline()
        while line:
            # print(line)
            parts = line.split(',')
            file_path = os.path.join(root, "audio", f"fold{parts[5]}", parts[0]).replace('\\', '/')
            if file_path[-3:] == "wav":
                X, sr = librosa.load(file_path, res_type='kaiser_fast')
                # print(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).shape)
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)
            print(mfccs)
            print(mfccs.shape)
            line = csvfile.readline()
            break


def audio_test():
    file_path = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold1/7061-6-0-0.wav"
    sample0, sr0 = soundfile.read(file_path)
    print(sample0.mean(axis=1).shape, sr0)
    sample1, sr1 = librosa.load(file_path)
    print(sample1.shape, sr1)


if __name__ == '__main__':
    # ext_list()
    create_file_list("metadata/UrbanSound8K.csv")
    # create_mfcc_npy_data("C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv")
    # audio_test()
    # create_mfcc_npy_data("C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/")
