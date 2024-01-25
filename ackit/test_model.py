# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-21 22:23
import yaml
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import torch.nn as nn

from ackit.models.tdnn import TDNN_Extractor
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


def test_tdnn_eval():
    wav_path = "D:/DATAS/Medical/COUGHVID-public_dataset_v3/coughvid_20211012/00d5c907-bae4-4747-971d-a231099c4bfa.wav"
    x_wav, sr = librosa.load(wav_path, sr=16000)
    tdnn_extractor = TDNN_Extractor().to(device)
    x_wav = torch.from_numpy(x_wav[np.newaxis, np.newaxis, :]).to(device)
    print(len(x_wav))
    x_feat = tdnn_extractor(x_wav)
    print(x_feat.shape)
    plt.figure(0)
    plt.imshow(x_feat.squeeze().data.cpu().numpy())
    plt.show()


if __name__ == '__main__':
    # test_vae()
    test_tdnn_eval()
    # test_mfn()
