# -*- coding: utf-8 -*-
# @Author : ZhaoKe
# @Time : 2024-01-21 22:23
import os
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import torch.nn as nn

from ackit.models.tdnn import TDNN_Extractor
from ackit.models.vae import ConvCVAE, vae_loss
from ackit.models.mobilefacenet import MobileFaceNet
from ackit.trainer_setting import get_model
from ackit.utils.utils import load_ckpt
from ackit.data_utils.featurizer import get_a_wavmel_sample

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


def test_vae_recon():
    cvae_model_path = "../runs/VAE/model_epoch_12"
    resume_model = cvae_model_path
    model = ConvCVAE(input_channel=1, input_length=288,
                     input_dim=128, latent_dim=8).to(device)
    load_epoch = 12
    load_ckpt(model, cvae_model_path, load_epoch=load_epoch)

    id_ma_map = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
    m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
    wav_path = [
        "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_fan/fan/train/",
        "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_pump/pump/train/",
        "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_slider/slider/train/",
        "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_valve/valve/train/",
        "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_ToyCar/ToyCar/train/",
        "G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_ToyConveyor/ToyConveyor/train/"
    ]

    # m_types = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
    # wav_path = [
    #     f"F:/DATAS/DCASE2024Task2ASD/dev_{item}/{item}/train/"
    #     for item in m_types
    # ]
    wav_list = [
        wav_path[i] + random.choice(os.listdir(wav_path[i])) for i in range(len(wav_path))
    ]
    num = len(m_types)
    for i in range(len(wav_list)):
        # if i != 3:
        #     continue
        _, x_input = get_a_wavmel_sample(wav_list[i])
        recons_spec, _, _, _ = model(x_input)#, torch.tensor([id_ma_map[m_types[i]]]))
        # print(recons_spec.shape)

        # print(z.shape)

        plt.figure(0)
        plt.subplot(2, num, i+1)
        plt.imshow(np.asarray(x_input.transpose(2, 3).squeeze().data.cpu().numpy()))

        plt.subplot(2, num, num+i+1)
        plt.imshow(np.asarray(recons_spec.transpose(2, 3).squeeze().data.cpu().numpy()))
        # plt.savefig(resume_model + f"testrecon_{load_epoch}_{m_types[i]}.png")
        # plt.close()
    plt.show()


if __name__ == '__main__':
    # test_vae()
    # test_tdnn_eval()
    # test_mfn()
    test_vae_recon()
