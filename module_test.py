#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/17 14:02
# @Author: ZhaoKe
# @File : module_test.py
# @Software: PyCharm
"""测试某些模块的正确性"""
from torch.utils.data import DataLoader

from ackit.utils.featurizer import AudioFeaturizer
from ackit.utils.reader import UrbansoundDataset, collate_fn_zero1_pad

def collate_and_mask_test():
    root = "C:/Program Files (zk)/data/UrbanSound8K/UrbanSound8K"
    train_list = "./datasets/train_list.txt"
    dataset = UrbansoundDataset(root=root, file_list=train_list, is_feat=False)
    dl = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_zero1_pad)
    af = AudioFeaturizer("MFCC")
    for idx, (wav, label, ilr) in enumerate(dl):
        feat = af(wav, ilr)
        print(feat)


if __name__ == '__main__':
    collate_and_mask_test()
