#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/14 11:13
# @Author: ZhaoKe
# @File : create_data.py
# @Software: PyCharm
import json
import os
import pickle

import numpy as np
import librosa
import pandas as pd
import soundfile

from ackit.utils.audio import vad, augment_audio, AudioSegment


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
                    valid_file.write(file_path + '\t' + parts[6] + '\n')
                elif ind % 10 == 9:
                    test_file.write(file_path + '\t' + parts[6] + '\n')
                else:
                    train_file.write(file_path + '\t' + parts[6] + '\n')
            line = csvfile.readline()
            ind += 1
            # break
    test_file.close()
    valid_file.close()
    train_file.close()


def pad_as_174(mfcc):
    max_len = 3000
    if mfcc.shape[1] < max_len:
        row_num, col_num = mfcc.shape
        new_mfcc = np.zeros((row_num, max_len))
        # print(new_mfcc.shape)
        pad_start = (max_len - col_num) // 2
        new_mfcc[:, pad_start:pad_start + col_num] = mfcc
        return new_mfcc.reshape(1, 40 * max_len)[0], pad_start
    else:
        # print(mfcc.shape)
        return mfcc.reshape(1, 40*max_len)[0], 0


def create_mfcc_npy_data(root, tra_val="train"):
    """
    train and valid 预先全部转换为MFCC存为npy文件,省去过多的IO时间,这样数据多在CPU和GPU之间传输,而非CPU和IO之间.
    问题在于数据太大，一次性保存到一个文件里，首先内存需要装得下，因此要考虑其他的存储方案，
    另外，为了找到最长的MFCC，还要先全部处理一编，应该先规定好数据的标准长度，这个要通过VAD之后的音频找到，以免多余静音数据太长，
    此外，只有array的话，可以分批次存储到多个npy里面。
    想到一个办法，把所有数据展为40*174的向量，然后写入到txt里面，需要的时候再reshape回来
    :param dataest_path:
    :return:
    @datetime: 2023-10-20 09:54 program block accomplished
    """
    is_vad, is_add_noise = True, True
    output_txt = open(f"./datasets/{tra_val}_vector.txt", 'w')
    # idx = 0
    with open(f"datasets/{tra_val}_list.txt", 'r') as csvfile:
        csvfile.readline()
        line = csvfile.readline()
        mfccs_to_save = []
        info_to_save = []
        idx = 0
        max_length = 0
        while line:
            # print(line)
            parts = line.strip().split('\t')
            file_path = os.path.join(root, parts[0]).replace('\\', '/')
            if file_path[-3:] == "wav":
                seg = AudioSegment.from_file(file_path)

                if is_vad:
                    if seg.samples.ndim > 1:
                        X = vad(seg.samples.mean(axis=1), top_db=20, overlap=200)
                    else:
                        X = vad(seg.samples, top_db=20, overlap=200)
                if is_add_noise:
                    X = augment_audio(noises_path=None,
                                      audio_segment=AudioSegment(X, seg.sample_rate),
                                      noise_dir="./datasets/noise")

                # print(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).shape)  # (40, 173)
                # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)
                # print(X.samples.shape)
                # print(X.samples.shape)
                mfccs = librosa.feature.mfcc(y=X.samples, sr=16000, n_mfcc=40)
                # if mfccs.ndim > 2:
                #     print(mfccs.shape)
                #     break
                # if mfccs.shape[1]>1500:
                #     print(mfccs.shape, parts[0])
                #     continue
                mfccs, pad_start = pad_as_174(mfccs)
                # if mfccs.ndim > 1:
                #     print(mfccs.shape)
                #     break
                # label data
                output_txt.write(parts[1] + ' ' + str(pad_start) + ' ' + ' '.join([str(x) for x in mfccs]) + ' ' + '\n')
                idx += 1
                if idx % 50 == 0:
                    print(f"already process waveform {idx}")
            #         mfccs_to_save.append(mfccs)
            #         info_to_save.append([int(parts[1]), 0])
            #         if mfccs.shape[1] > max_length:
            #             max_length = mfccs.shape[1]
            #     # print(mfccs)
            #     # print(mfccs.shape)
            line = csvfile.readline()
        #     # break
        #     if idx % 200 == 0 and idx > 1:
        #         print(f"already process waveform {idx}")
        #         # break
        #     idx += 1
        #     # break
        #
        # # (40, 173), 173并非固定
        # # print("--------padding----------")
        # print("max_len: ", max_length)
        # print(len(mfccs_to_save))
        # for i, mfcc in enumerate(mfccs_to_save):
        #     if mfcc.shape[1] < max_length:
        #         row_num, col_num = mfcc.shape
        #         new_mfcc = np.zeros((row_num, max_length))
        #         # print(new_mfcc.shape)
        #         pad_start = (max_length - col_num) // 2
        #         new_mfcc[:, pad_start:pad_start + col_num] = mfcc
        #         mfccs_to_save[i] = new_mfcc
        #         info_to_save[i][1] = pad_start
        # np.save(f"./data_entities/{tra_val}_mfcc_vad_noise.npy", mfccs_to_save)
        # np.save(f"./data_entities/{tra_val}_info_vad_noise.npy", info_to_save)
    output_txt.close()


def audio_test():
    file_path = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold2/17307-1-0-0.wav"
    # sample0, sr0 = soundfile.read(file_path)
    # print(sample0.mean(axis=1).shape, sr0)
    sample1, sr1 = librosa.load(file_path)
    print(sample1.shape, sr1)
    file_path = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold2/156893-7-9-0.wav"
    sample2, sr2 = librosa.load(file_path)
    print(sample2.shape, sr2)
    file_path = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold2/147926-0-0-28.wav"
    sample3, sr3 = librosa.load(file_path)
    print(sample3.shape, sr3)


def read_npy_test():
    train_mfccs = np.load("./datasets/valid_mfcc.npy")
    print(len(train_mfccs))
    for i in range(10):
        print(train_mfccs[i].shape)
        print(train_mfccs[1])
    train_mfccs = np.load("./datasets/valid_info.npy")
    print(train_mfccs)


def detect_short():
    root = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio"
    with open("datasets/train_list.txt", 'r') as csvfile:
        csvfile.readline()
        line = csvfile.readline()
        # mfccs_to_save = []
        # idx = 0
        # max_length = 0
        while line:
            # print(line)
            parts = line.split('\t')
            file_path = os.path.join(root, parts[0]).replace('\\', '/')
            if file_path[-3:] == "wav":
                X, sr = librosa.load(file_path, res_type='kaiser_fast')
                if X.shape[-1] < 2048:
                    print(parts[0])


if __name__ == '__main__':
    # ext_list()
    # create_file_list("metadata/UrbanSound8K.csv")
    create_mfcc_npy_data("C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio", tra_val="train")
    # create_mfcc_npy_data("C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio", tra_val="valid")
    # print("-------------test-------------------")
    # create_mfcc_npy_data("C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio", tra_val="test")
    # read_npy_test()
    # audio_test()
    # detect_short()
    # create_mfcc_npy_data("C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/")
