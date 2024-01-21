# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-21 22:23
import yaml
import torch
import torch.nn as nn
from ackit.models.vae import ConvVAE, vae_loss
from ackit.models.mobilefacenet import MobileFaceNet
from ackit.trainer_setting import get_model

rec_loss = nn.MSELoss()
cls_loss = nn.CrossEntropyLoss()
configs_path = "../configs/autoencoder.yaml"
with open(configs_path) as stream:
    configs = yaml.safe_load(stream)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
input_mel = torch.rand(64, 1, 288, 128).to(device)
input_mid = torch.randint(0, 23, size=(64,)).to(device)

def test_vae():
    VAE_model = get_model(use_model="vae", configs=configs, istrain=True).to(device)
    recon_mel, latent_feat, latent_mean, latent_logvar = VAE_model(input_mel)
    print("output:")
    print(recon_mel.shape)
    print(latent_feat.shape)
    print(latent_mean.shape)
    print(latent_logvar.shape)
    loss_value = vae_loss(recon_mel, input_mel, latent_mean, latent_logvar)
    loss_value.backward()
    print(loss_value)


def test_mfn():
    mfn_model = MobileFaceNet(inp_c=1, num_class=23)
    pred, feat = mfn_model(x=input_mel.transpose(2, 3), label=input_mid)
    print("output:")
    print(pred.shape)
    print(feat.shape)


if __name__ == '__main__':
    test_vae()
    # test_mfn()
