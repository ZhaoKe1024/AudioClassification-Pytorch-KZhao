#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/17 14:02
# @Author: ZhaoKe
# @File : module_test.py
# @Software: PyCharm
"""测试某些模块的正确性"""
import os

import soundfile
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from ackit.utils.audio import vad, augment_audio, AudioSegment
from ackit.utils.featurizer import AudioFeaturizer
from ackit.data_utils.reader import UrbansoundDataset, collate_fn_zero1_pad


def collate_and_mask_test():
    root = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K"
    train_list = "./data_entities/train_list.txt"
    dataset = UrbansoundDataset(root=root, file_list=train_list, is_feat=False)
    dl = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_zero1_pad)
    af = AudioFeaturizer("MFCC")
    for idx, (wav, label, ilr) in enumerate(dl):
        feat = af(wav, ilr)
        print(feat)


def dataset_test():
    ds = UrbansoundDataset("C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/", "train", is_feat=True)
    dl = DataLoader(ds,
                    batch_size=16,
                    shuffle=True, num_workers=0)
    # af = AudioFeaturizer(feature_method="MFCC")
    for idx, (data, label, dlr) in enumerate(dl):
        print(idx, '\t', data.shape, '\t', len(dlr))
        # x = torch.tensor(data)
        # feat = af(data, dlr)
        print(data.shape, dlr)
    # sample1, sr1 = ds[1]
    # sample2, sr2 = ds[2]
    # print(sample1.shape)
    # print(sample2.shape)


def add_audio_test():
    wav_file = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold8/36429-2-0-6.wav"
    X_segment = AudioSegment.from_file(wav_file)
    print("X_segment: ", X_segment.samples.shape)
    X_vaded = vad(X_segment.samples, top_db=20, overlap=200)
    print("X_vaded: ", X_vaded.shape)
    x_noise = augment_audio(noises_path=None,
                            audio_segment=AudioSegment(X_vaded, X_segment.sample_rate),
                            noise_dir="./datasets/noise")
    print("X_noise: ", x_noise.samples.shape)

    diff = X_vaded - x_noise.samples
    print(sum(diff))
    plt.figure(0)
    plt.subplot(3, 1, 1)
    plt.plot(X_segment.samples, c='blue')
    # plt.plot(X_segment.samples[:, 1], c='red')
    plt.subplot(3, 1, 2)

    # plt.subplot(3,1,3)
    plt.plot(x_noise.samples, c='green')
    # plt.xlim([0, len(X_segment.samples)])
    plt.plot(X_vaded, c='blue')
    plt.xlim([0, len(X_segment.samples)])
    # plt.plot(X_vaded[:,1], c='red')
    plt.show()


def add_audio_play_test():
    wav_file = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold8/36429-2-0-6.wav"
    X_segment = AudioSegment.from_file(wav_file)
    print("X_segment: ", X_segment.samples.shape)
    X_vaded = vad(X_segment.samples, top_db=20, overlap=200)
    print("X_vaded: ", X_vaded.shape)
    x_noise = augment_audio(noises_path=None,
                            audio_segment=AudioSegment(X_vaded, X_segment.sample_rate),
                            noise_dir="./datasets/noise")
    print("X_noise: ", x_noise.samples.shape)
    soundfile.write("datasets/test_audio/36429-2-0-6-noise.wav", x_noise.samples, x_noise.sample_rate)
    diff = X_vaded - x_noise.samples
    print(sum(diff))


def plot_a_wav():
    wav_file = "./datasets/test_audio/36429-2-0-6-noise.wav"
    wav_seg = AudioSegment.from_file(wav_file)
    wav_file_org = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K/audio/fold8/36429-2-0-6.wav"
    wav_seg_org = AudioSegment.from_file(wav_file_org)
    plt.figure(0)
    plt.plot(wav_seg.samples, c='blue')
    plt.plot(wav_seg_org.samples, c='red')
    plt.show()


def stats_noise_track():
    s_dcit = {1: 0}
    for item in os.listdir("./datasets/noise"):
        sample, sr = soundfile.read("./data_entities/noise"+'/'+item)
        if sample.ndim > 1:
            print(sample.shape)
            if sample.shape[1] in s_dcit:
                s_dcit[sample.shape[1]] += 1
            else:
                s_dcit[sample.shape[1]] = 1
                print(s_dcit)
        else:
            s_dcit[1] += 1
    print("=========")
    print(s_dcit)


if __name__ == '__main__':
    # dataset_test()
    stats_noise_track()
    # a = torch.randn((3, 3))
    # for i in range(4):
    # add_audio_test()
    # plot_a_wav()
    # add_audio_play_test()
    # b = torch.zeros((3, 8))
    # b[:, 2:5] = a
    # print(b)

    # collate_and_mask_test()
