#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/16 19:06
# @Author: ZhaoKe
# @File : autoencoder.py
# @Software: PyCharm
import torch.nn as nn
from ackit.modules.loss import ArcMarginProduct


class ConvEncoder(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128, class_num=23, class_num1=6):
        super(ConvEncoder, self).__init__()
        self.input_dim = input_channel
        self.max_cha = 256
        es = [input_channel, 32, 64, 128, self.max_cha]  # , 128]
        self.encoder_layers = nn.Sequential()
        kernel_size, stride, padding = 4, 2, 1
        for i in range(len(es) - 2):
            self.encoder_layers.append(
                nn.Conv2d(es[i], es[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.LayerNorm((es[i + 1], input_length // (2 ** (i + 1)), input_dim // (2 ** (i + 1)))))
            self.encoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.encoder_layers.append(nn.Conv2d(es[-2], es[-1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        # z_len = input_length // 8
        # z_dim = input_dim // 8
        self.class_num23 = class_num
        self.arcface = ArcMarginProduct(in_features=input_dim * 2, out_features=class_num)
        self.fc_out = nn.Linear(in_features=input_dim * 2, out_features=class_num1)

    def forward(self, input_mel, class_vec, coarse_cls=True, fine_cls=False):
        z = self.encoder_layers(input_mel)
        if fine_cls:
            latent_pred = self.arcface(z.mean(axis=3).mean(axis=2), class_vec)
            if coarse_cls:
                coarse_pred = self.fc_out(z.mean(axis=3).mean(axis=2))
                return z, coarse_pred, latent_pred
            return z, latent_pred
        elif coarse_cls:
            coarse_pred = self.fc_out(z.mean(axis=3).mean(axis=2))
            return z, coarse_pred
        else:
            return z


class ConvDecoder(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128):
        super(ConvDecoder, self).__init__()
        self.max_cha = 256
        kernel_size, stride, padding = 4, 2, 1
        z_len = input_length // 8
        z_dim = input_dim // 8
        ds = [self.max_cha, 128, 64, 32, input_channel]

        self.decoder_layers = nn.Sequential()
        self.decoder_layers.append(
            nn.ConvTranspose2d(ds[0], ds[1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        self.decoder_layers.append(nn.LayerNorm((ds[1], z_len, z_dim)))
        self.decoder_layers.append(nn.ReLU(inplace=True))
        for i in range(1, len(ds) - 2):
            self.decoder_layers.append(
                nn.ConvTranspose2d(ds[i], ds[i + 1], kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False))
            self.decoder_layers.append(nn.LayerNorm((ds[i + 1], z_len * 2 ** i, z_dim * 2 ** i)))
            self.decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder_layers.append(
            nn.ConvTranspose2d(ds[-2], ds[-1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.decoder_layers.append(nn.Tanh())

    def forward(self, latent_map):
        """
        :param latent_map: shape (33, 13)
        :return:
        """
        d = self.decoder_layers(latent_map)
        return d
