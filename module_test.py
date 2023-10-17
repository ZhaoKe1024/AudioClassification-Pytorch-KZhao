#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/17 14:02
# @Author: ZhaoKe
# @File : module_test.py
# @Software: PyCharm
"""测试某些模块的正确性"""
import yaml
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


if __name__ == '__main__':
    dataset_test()
    # a = torch.randn((3, 3))
    # b = torch.zeros((3, 8))
    # b[:, 2:5] = a
    # print(b)

    # collate_and_mask_test()
