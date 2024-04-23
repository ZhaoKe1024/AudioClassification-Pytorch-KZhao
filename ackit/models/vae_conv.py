#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/2/27 20:00
# @Author: ZhaoKe
# @File : vae_conv.py
# @Software: PyCharm
from collections import deque, OrderedDict
import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    def __init__(self, shape=(1, 288, 128), hidden_dim=16):
        super(ConvVAE, self).__init__()
        print("Load New VAE Model")
        c, h, w = shape
        hh, ww = h, w
        self.shapes = [(hh, ww)]

        self.img_reduce = Reduction4dto2d(shape=shape)
        cc = self.img_reduce.cc

        self.calc_mean = MLP([128, 64, hidden_dim], last_activation=False)
        self.calc_logvar = MLP([128, 64, hidden_dim], last_activation=False)

        self.decoder_lin = MLP([hidden_dim, 64, 128, 256, cc * self.img_reduce.ww * self.img_reduce.hh])
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(cc, self.img_reduce.hh, self.img_reduce.ww))

        self.maxunpool2d1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.decoder_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0)
        self.decoder_norm1 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.decoder_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.decoder_norm2 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.maxunpool2d2 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.decoder_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=(2, 1), padding=0)
        self.decoder_norm3 = nn.Sequential(nn.BatchNorm2d(16), nn.ReLU(inplace=True))

        self.decoder_conv4 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=0)

        # self.decoder = Decoder(shape, hidden_dim, ncond=cond_dim)

    def sampling(self, mean, logvar, device=torch.device("cuda")):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        x_feat, indices = self.img_reduce(x)
        # print("x_latent enc mlp:", x_feat.shape)

        mean_lant, logvar_lant = self.calc_mean(x_feat), self.calc_logvar(x_feat)
        z = self.sampling(mean_lant, logvar_lant, device=torch.device("cuda"))
        # print("mean logvar z:", mean_lant.shape, logvar_lant.shape, z.shape)

        x_feat = self.decoder_lin(z)
        # print("x_latent dec lin:", x_feat.shape)
        x_feat = self.unflatten(x_feat)
        # print("x_latent unflatten:", x_feat.shape)

        x_recon = self.maxunpool2d1(x_feat, indices=indices.pop(), output_size=self.img_reduce.shapes[-2])
        # print("recon_x unpool1:", x_recon.shape)

        x_recon = self.decoder_conv1(x_recon, output_size=self.img_reduce.shapes[-3])
        # print("recon_x:", x_recon.shape)
        x_recon = self.decoder_norm1(x_recon)

        x_recon = self.decoder_conv2(x_recon, output_size=self.img_reduce.shapes[-4])
        # print("recon_x:", x_recon.shape)
        x_recon = self.decoder_norm2(x_recon)

        x_recon = self.maxunpool2d2(x_recon, indices=indices.pop(), output_size=self.img_reduce.shapes[-5])
        # print("unpool_2:", x_recon.shape)

        x_recon = self.decoder_conv3(x_recon, output_size=self.img_reduce.shapes[-6])
        # print("recon_x:", x_recon.shape)
        x_recon = self.decoder_norm3(x_recon)

        x_recon = self.decoder_conv4(x_recon, output_size=self.img_reduce.shapes[-7])
        # print("recon_x:", x_recon.shape)
        return x_recon, mean_lant, logvar_lant

    def generate(self, batch_size=None, labels=None, device=torch.device("cuda")):
        z = torch.randn((batch_size, self.dim)).to(device) if batch_size else torch.randn((1, self.dim)).to(device)
        if not self.iscond:
            res = self.decoder(z)
        else:
            res = self.decoder(z, labels)
        if not batch_size:
            res = res.squeeze(0)
        return res


MSE_loss = nn.MSELoss(reduction="mean")


def vae_loss(X, X_hat, mean, logvar, kl_weight=0.0001):
    reconstruction_loss = MSE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    # print(reconstruction_loss.item(), KL_divergence.item())
    return reconstruction_loss + kl_weight * KL_divergence


class Reduction4dto2d(nn.Module):
    def __init__(self, shape=(1, 288, 128), hidden_dim=128, flatten=True):
        super().__init__()
        c, h, w = shape
        hh, ww = h, w
        self.shapes = [(hh, ww)]

        self.encoder_conv1 = nn.Sequential()

        self.encoder_conv1.append(nn.Conv2d(c, 16, kernel_size=5, stride=2, padding=0))
        self.encoder_conv1.append(nn.BatchNorm2d(16))
        self.encoder_conv1.append(nn.ReLU(inplace=True))
        hh = int((hh - 5 + 2 * 0) / 2) + 1
        ww = int((ww - 5 + 2 * 0) / 2) + 1
        self.shapes.append((hh, ww))
        self.encoder_conv1.append(nn.Conv2d(16, 32, kernel_size=5, stride=(2, 1), padding=0))
        self.encoder_conv1.append(nn.BatchNorm2d(32))
        self.encoder_conv1.append(nn.ReLU(inplace=True))
        hh = int((hh - 5 + 2 * 0) / 2) + 1
        ww = int((ww - 5 + 2 * 0) / 1) + 1
        self.shapes.append((hh, ww))

        self.maxpool2d1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        hh, ww = hh // 2, ww // 2
        self.shapes.append((hh, ww))

        self.encoder_conv2 = nn.Sequential()
        self.encoder_conv2.append(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0))
        self.encoder_conv2.append(nn.BatchNorm2d(64))
        self.encoder_conv2.append(nn.ReLU(inplace=True))
        hh = (hh - 3 + 2 * 0) // 1 + 1
        ww = (ww - 3 + 2 * 0) // 1 + 1
        self.shapes.append((hh, ww))

        self.encoder_conv2.append(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0))
        self.encoder_conv2.append(nn.BatchNorm2d(128))
        self.encoder_conv2.append(nn.ReLU(inplace=True))
        hh = (hh - 3 + 2 * 0) // 2 + 1
        ww = (ww - 3 + 2 * 0) // 2 + 1
        self.shapes.append((hh, ww))

        self.maxpool2d2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        hh, ww = hh // 2, ww // 2
        self.shapes.append((hh, ww))

        self.cc = 128
        self.hh, self.ww = self.shapes[-1]
        # print("out:", (cc, hh, ww))
        self.flatten = flatten
        if flatten:
            self.encoder_mlp = nn.Sequential(Flatten(), MLP([ww * hh * self.cc, 256, hidden_dim]))

    def forward(self, x):
        indices = deque()
        x_feat = self.encoder_conv1(x)
        # print("x_feat conv1:", x_feat.shape)
        x_feat, index1 = self.maxpool2d1(x_feat)
        # print("x_feat maxpool1:", x_feat.shape)
        x_feat = self.encoder_conv2(x_feat)
        # print("x_feat conv2:", x_feat.shape)
        x_feat, index2 = self.maxpool2d2(x_feat)
        # print("x_feat maxpool2:", x_feat.shape)
        if self.flatten:
            x_feat = self.encoder_mlp(x_feat)
        # print("x_latent enc mlp:", x_feat.shape)
        indices.append(index1)
        indices.append(index2)
        return x_feat, indices


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


def prod(input_list):
    res = 1
    for item in input_list:
        res *= item
    return res


def test_mse():
    import torch.nn.functional as F
    x_spec = torch.rand(32, 1, 288, 128)  # .to("cuda")
    feature = torch.rand(32, 1, 288, 128)  # .to("cuda")
    loss_fn = nn.MSELoss()
    print(loss_fn(x_spec, feature))
    print(F.mse_loss(x_spec, feature))
    print(F.mse_loss(x_spec, feature, reduction="sum"))
    print(F.mse_loss(x_spec, feature, reduction="sum") / prod(list(x_spec.shape)))
    print(F.mse_loss(x_spec, feature, reduction="mean"))
    print(F.mse_loss(x_spec, feature, reduction="none").shape)
    print(F.mse_loss(x_spec, feature, reduction="none").mean(axis=3).mean(axis=2).mean(axis=1).shape)


if __name__ == '__main__':
    from modules.func import onehotvector
    import torch.nn.functional as F
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    bs, classnum = 16, 7
    # vaeconv = ConvVAE(shape=(1, 288, 128)).to(device)
    encoder = Reduction4dto2d(shape=(1, 288, 128)).to(device)
    # print(vaeconv)
    input_img = torch.randn(size=(bs, 1, 288, 128)).to(device)
    input_y = onehotvector(torch.randint(0, classnum, size=(bs, 1)), classnum)
    print("input_y:\n", input_y)
    # decoded, m, lo = vaeconv(input_img)
    latent_vec, _ = encoder(input_img)
    print(latent_vec.shape)
    # print(input_img.shape, decoded.shape)
    # print(input_img)
    # print(decoded)
    # loss = vae_loss(input_img, decoded, m, lo)

    # print(loss)

    # loss.backward()
    # vaeconv = VAE(shape=(1, 28, 28), nhid=32, cond_dim=32)
    # print(vaeconv)