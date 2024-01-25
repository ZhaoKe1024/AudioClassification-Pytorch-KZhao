# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-26 0:53
import random
import numpy as np


def wav_slice_padding(old_signal, save_len=160000):
    new_signal = np.zeros(save_len)
    if old_signal.shape[0] < save_len:
        resi = save_len - old_signal.shape[0]
        # print("resi:", resi)
        new_signal[:old_signal.shape[0]] = old_signal
        new_signal[old_signal.shape[0]:] = old_signal[-resi:][::-1]
    elif old_signal.shape[0] > save_len:
        posi = random.randint(0, old_signal.shape[0] - save_len)
        new_signal = old_signal[posi:posi+save_len]
    return new_signal
