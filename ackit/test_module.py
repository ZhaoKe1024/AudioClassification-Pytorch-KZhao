# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-22 22:02
import random
import numpy as np
import torch
import torch.nn as nn


x_mel = torch.randn(size=(64, 1, 288, 128))
x_mtid = torch.randint(0, 23, size=(64,))
x_ytrue = torch.zeros(size=(64,))


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margim = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, pos, neg):
        part1 = (anchor - pos).pow(2).sum(dim=1)
        part2 = (anchor - neg).pow(2).sum(dim=1)
        return self.relu(part1 - part2 + self.margim).mean()


def test_cls_triple():
    # batch_size, c, h, w = x_mel.shape
    # sample_num = batch_size // 4
    # indices_chosen = random.choices(list(range(batch_size)), k=sample_num*2)
    # x_ytrue[indices_chosen[sample_num:], :, ...] = 1
    # mtid_pred_loss = torch.randn(64, )
    latent_h = torch.randn(64, 128)
    latent_p = torch.randn(64, 128)
    latent_n = torch.randn(64, 128)
    closs = ContrastiveLoss()
    contrast_loss = closs(latent_h, latent_p, latent_n)
    contrast_loss.backward()


if __name__ == '__main__':
    test_cls_triple()
