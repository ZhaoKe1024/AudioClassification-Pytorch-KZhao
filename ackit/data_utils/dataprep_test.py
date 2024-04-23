#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/23 15:19
# @Author: ZhaoKe
# @File : dataprep_test.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import librosa
from ackit.data_utils.audio import AudioSegment

root_path = "F:/DATAS/COUGHVID-public_dataset_v3/coughvid_20211012/"


def vad_plot():
    audio1 = AudioSegment.from_file(root_path + "ffe8b243-9c7b-49f1-8d7d-a953f736ea4b.wav")
    print("sample rate:", audio1.sample_rate)
    plt.figure(0)
    plt.subplot(2, 1, 1)
    plt.plot(range(len(audio1.samples)), audio1.samples, color="black")
    audio1.vad()
    plt.subplot(2, 1, 2)
    print(len(audio1.samples))
    print(audio1.duration)
    plt.plot(range(len(audio1.samples)), audio1.samples, color="black")
    plt.show()


if __name__ == '__main__':
    vad_plot()
