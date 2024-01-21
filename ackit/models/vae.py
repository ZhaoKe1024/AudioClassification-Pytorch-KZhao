# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-21 22:18
import torch
import torch.nn as nn
import torch.nn.functional as F

from ackit.models.autoencoder import ConvDecoder


class ConvVAE(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128):  # , class_num=23, class_num1=6):
        super(ConvVAE, self).__init__()
        self.input_dim = input_channel
        self.max_cha = 256
        es = [input_channel, 32, 64, 128, self.max_cha]  # , 128]
        self.encoder_layers = nn.Sequential()
        kernel_size, stride, padding = 4, 2, 1

        for i in range(len(es) - 2):
            self.encoder_layers.append(nn.Conv2d(es[i], es[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.LayerNorm((es[i + 1], input_length // (2 ** (i + 1)), input_dim // (2 ** (i + 1)))))
            self.encoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.encoder_layers.append(nn.Conv2d(es[-2], es[-1], kernel_size=kernel_size, stride=1, padding=0, bias=False))

        self.z_len = input_length // 8 - 3
        self.z_dim = input_dim // 8 - 3
        # print("z:", self.z_len, self.z_dim)
        self.mean_linear = nn.Linear(self.max_cha * self.z_len * self.z_dim,
                                     self.max_cha)
        self.var_linear = nn.Linear(self.max_cha * self.z_len * self.z_dim,
                                    self.max_cha)
        # self.latent_dim = self.max_cha
        # self.class_num23 = class_num
        # self.arcface = ArcMarginProduct(in_features=input_dim * 2, out_features=class_num)
        # self.fc_out = nn.Linear(in_features=input_dim * 2, out_features=class_num1)

        self.decoder_projection = nn.Linear(self.max_cha, self.max_cha * self.z_len * self.z_dim)
        self.decoder_layers = ConvDecoder(input_channel=input_channel, input_length=input_length, input_dim=input_dim)

    def forward(self, input_mel):
        # print("input mel:", input_mel.shape)
        z = self.encoder_layers(input_mel)
        print("encoder output:", z.shape)
        encoded = torch.flatten(z, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = eps + std + mean
        decode_input = self.decoder_projection(z)
        # print("decode input:", decode_input.shape)
        decode_input = torch.reshape(decode_input, (-1, self.max_cha, self.z_len, self.z_dim))
        # print("input decoder:", decode_input.shape)
        decoded = self.decoder_layers(decode_input)
        # print("decode output:", decoded.shape)
        return decoded, z, mean, logvar


def vae_loss(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar), 1), 0)
    kl_weight = 0.00025
    loss = recons_loss + kl_loss * kl_weight
    return loss
