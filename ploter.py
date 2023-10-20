#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/18 10:24
# @Author: ZhaoKe
# @File : ploter.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_line(txt_file_path, topic="accuracy"):
    print(txt_file_path.split('.')[0].split('/')[-1])
    data = np.loadtxt(txt_file_path, delimiter=',', dtype=np.float32)
    # print(len(data))
    # print(data)
    x_range = range(len(data))
    plt.figure(0)
    plt.plot(x_range, data, c='black')
    plt.xticks(x_range, [f"epoch-{i}" for i in x_range], rotation=67)
    plt.xlabel("epoch")
    plt.ylabel(topic)
    plt.title(txt_file_path.split('.')[0].split('/')[-1])
    plt.grid('on')
    plt.show()


def plot_accuracy_2line(txt_filelist, topic="accuracy"):
    plt.figure(0)
    data = []
    for filepath in txt_filelist:
        data.append(np.loadtxt(filepath, delimiter=',', dtype=np.float32))
    labels = ["v0.1 read wav", "v0.2 read mfcc"]
    colors = ["blue", "red", "green", "orange"]
    # print(len(data))
    # print(data)
    x_range = range(len(data[0]))
    for i in range(len(data)):
        plt.plot(x_range, data[i], c=colors[i], label=labels[i])

    plt.xticks(x_range, [f"epoch-{i}" for i in x_range], rotation=67)
    plt.xlabel("epoch")
    plt.ylabel(topic)
    # plt.title(txt_file_path.split('.')[0].split('/')[-1])
    plt.legend()
    plt.grid('on')
    plt.show()

if __name__ == '__main__':
    file_path_list = [
        "./runs/tdnn_MFCC_mfccvector/train_acc_epoch-20.txt",
        "./runs/tdnn_MFCC/train_acc_epoch-20.txt"
    ]
    plot_accuracy_2line(file_path_list, "accuracy")
