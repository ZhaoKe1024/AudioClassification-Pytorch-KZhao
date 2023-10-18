#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/18 10:24
# @Author: ZhaoKe
# @File : ploter.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_line(txt_file_path, topic="accuracy"):
    data = np.loadtxt(txt_file_path, delimiter=',', dtype=np.float32)
    # print(len(data))
    # print(data)
    x_range = range(len(data))
    plt.figure(0)
    plt.plot(x_range, data, c='black')
    plt.xticks(x_range, [f"epoch-{i}" for i in x_range], rotation=67)
    plt.xlabel("epoch")
    plt.ylabel(topic)
    plt.title(txt_file_path.split('.')[0])
    plt.grid('on')
    plt.show()


if __name__ == '__main__':
    plot_accuracy_line("./runs/tdnn_MFCC/train_acc_epoch-20.txt", "accuracy")
