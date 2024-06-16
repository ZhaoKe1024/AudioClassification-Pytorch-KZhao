#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/6/16 12:39
# @Author: ZhaoKe
# @File : dcase20_w2v.py
# @Software: PyCharm
import os
import math
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import librosa
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ackit.pretrained.wav2vec import Wav2Vec
from ackit.models.mobilenetv2 import MobileNetV2
# from ackit.modules.classifiers import LSTM_Classifier, LSTM_Attn_Classifier
from readers.featurizer import wav_slice_padding


class WaveReader(Dataset):
    def __init__(self, file_paths, mtype_list, mtid_list):
        self.files = file_paths
        self.mtype_list = mtype_list
        self.mtids = mtid_list
        self.wav_list = []
        for fi in tqdm(file_paths, desc=f"build Set..."):
            y, sr = librosa.core.load(fi, sr=16000)
            y = wav_slice_padding(y, save_len=16000)
            self.wav_list.append(y)

    def __getitem__(self, ind):
        return self.wav_list[ind], self.mtype_list[ind], self.mtids[ind]

    def __len__(self):
        return len(self.files)


def get_type_loader(configs=None, m2l=None):
    # generate dataset
    # generate dataset
    print("============== DATASET_GENERATOR ==============")
    ma_id_map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
    every_num, every_valid_num = 1200, 200
    cnts_train, cnts_valid = [0] * 6, [0] * 6
    print("---------------train dataset-------------")
    file_train, file_valid = [], []
    mtype_train, mtype_valid = [], []
    mtid_train, mtid_valid = [], []
    with open("../datasets/d2020_trainlist.txt", 'r') as fin:
        train_path_list = fin.readlines()
        for item in train_path_list:
            parts = item.strip().split('\t')
            machine_type_id = int(parts[1])
            machine_id_id = parts[2]
            meta = ma_id_map[machine_type_id] + '-id_' + machine_id_id
            if cnts_train[machine_type_id] < every_num:
                file_train.append(parts[0])
                mtype_train.append(machine_type_id)
                mtid_train.append(m2l[meta])
                cnts_train[machine_type_id] += 1
            elif cnts_valid[machine_type_id] < every_valid_num:
                file_valid.append(parts[0])
                mtype_valid.append(machine_type_id)
                mtid_valid.append(m2l[meta])
                cnts_valid[machine_type_id] += 1
            else:
                continue
    train_dataset = WaveReader(file_paths=file_train, mtype_list=mtype_train, mtid_list=mtid_train)
    train_loader = DataLoader(train_dataset, batch_size=configs["fit"]["batch_size"], shuffle=True)
    test_dataset = WaveReader(file_paths=file_valid, mtype_list=mtype_valid, mtid_list=mtid_valid)
    test_loader = DataLoader(test_dataset, batch_size=configs["fit"]["batch_size"], shuffle=False)
    return train_loader, test_loader


def train():
    timestr = time.strftime("%Y%m%d%H%M", time.localtime())
    run_save_dir = "../runs/dcase20cls" + '/' + timestr + f'w2n7c/'
    configs = {
        "fit": {
            "batch_size": 32,
        },
        "feature": {
            "wav_length": 20865
        }
    }
    setting_content = "Wave2Vector, wav_length: 16000, batch_size: 32, num_type: 6; \n"
    setting_content += "iter_max = 50; warm_up_iter, T_max, lr_max, lr_min = 10, iter_max // 1.5, 5e-3, 5e-4;"
    setting_content += "LSTM_Attn_Classifier(inp_size=512, hidden_size=64, n_classes=6, return_attn_weights=False, attn_type='dot');"
    setting_content += "optimizer = optim.Adam(cl_model.parameters(), lr=5e-4);"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    with open("../datasets/d2020_metadata2label.json", 'r', encoding='utf_8') as fp:
        meta2label = json.load(fp)
    train_loader, test_loader = get_type_loader(configs=configs, m2l=meta2label)
    w2v = Wav2Vec(pretrained=True, pretrained_path="../ackit/pretrained/wav2vec_large.pt")
    cl_model = MobileNetV2(dc=1, n_class=6, input_size=32)
    # cl_model = LSTM_Attn_Classifier(inp_size=512, hidden_size=64, n_classes=6, return_attn_weights=False, attn_type='dot')
    criterion = nn.CrossEntropyLoss()

    iter_max = 50
    warm_up_iter, T_max, lr_max, lr_min = 10, iter_max // 1, 5e-3, 5e-4
    # reference: https://blog.csdn.net/qq_36560894/article/details/114004799
    # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
    lambda0 = lambda cur_iter: lr_max * cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
        (lr_min + 0.5 * (lr_max - lr_min) * (
                1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1
    optimizer = optim.Adam(cl_model.parameters(), lr=lr_max)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=5e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

    w2v, cl_model, criterion = w2v.to(device=device), cl_model.to(device=device), criterion.to(device)

    old = 0
    STD_acc = []
    STD_loss = []
    loss_line = []
    lr_list = []

    for epoch_id in range(iter_max):
        print("Epoch", epoch_id)
        cl_model.train()
        loss_list = []
        lr_list.append(optimizer.param_groups[0]['lr'])
        for idx, (wav, mtp, _) in enumerate(tqdm(train_loader, desc="Train")):
            wav = wav.to(device)
            mtp = mtp.to(device)
            mel = w2v(wav)
            optimizer.zero_grad()
            pred = cl_model(x=mel.unsqueeze(1))
            loss_v = criterion(pred, mtp)
            # torch.Size([32, 16000]) torch.Size([32, 512, 98]) torch.Size([32, 6]) torch.Size([])
            # print("shape wav, mel, pred:", wav.shape, mel.shape, pred.shape, loss.shape)
            loss_list.append(loss_v.item())
            loss_v.backward()

            optimizer.step()
            # if idx > 10:
            #     break
        loss_line.append(np.array(loss_list).mean())
        cl_model.eval()
        with torch.no_grad():
            acc_list = []
            loss_list = []
            for idx, (wav, mtp, _) in enumerate(tqdm(test_loader, desc="Test")):
                wav = wav.to(device)
                mtp = mtp.to(device)
                mel = w2v(wav)
                pred = cl_model(x=mel.unsqueeze(1))
                loss_eval = criterion(pred, mtp)
                acc_batch = metrics.accuracy_score(mtp.data.cpu().numpy(),
                                                   pred.argmax(-1).data.cpu().numpy())
                acc_list.append(acc_batch)
                loss_list.append(loss_eval.item())
            acc_per = np.array(acc_list).mean()
            # print("new acc:", acc_per)
            STD_acc.append(acc_per)
            STD_loss.append(np.array(loss_list).mean())
            if acc_per > old:
                old = acc_per
                print(f"Epoch[{epoch_id}] new acc: {acc_per}")
                if acc_per > 0.85:
                    print(f"Epoch[{epoch_id}]: {acc_per}")
                    if not os.path.exists(run_save_dir):
                        os.makedirs(run_save_dir, exist_ok=True)
                        with open(run_save_dir + f"setting.txt", 'w') as fin:
                            # fin.write("MobileNetV2, adam cosine anneal 5e-4 ~ 5e-5, data augmentation, feature map max reduction.")
                            fin.write(setting_content)
                    torch.save(cl_model.state_dict(), run_save_dir + f"cls_model_{epoch_id}.pt")
                    torch.save(optimizer.state_dict(), run_save_dir + f"optimizer_{epoch_id}.pt")
        scheduler.step()
    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.plot(range(len(loss_line)), loss_line, c="red", label="train_loss")
    plt.plot(range(len(STD_loss)), STD_loss, c="blue", label="valid_loss")
    plt.plot(range(len(STD_acc)), STD_acc, c="green", label="valid_accuracy")
    plt.xlabel("iteration")
    plt.ylabel("metrics")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(lr_list)
    if not os.path.exists(run_save_dir):
        os.makedirs(run_save_dir, exist_ok=True)
        with open(run_save_dir + f"setting.txt", 'w') as fin:
            fin.write(setting_content)
    plt.savefig(run_save_dir + "train_result.png", format="png", dpi=300)
    plt.show()


def heatmap_eval():
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # cl_model = MobileNetV2(dc=1, n_class=5, input_size=288, width_mult=1).to(device)
    cl_model = LSTM_Classifier(inp_size=512, hidden_size=64, n_classes=6).to(device)
    cl_model.load_state_dict(torch.load(f"../runs/dcase20cls/202406161503w2n7c/cls_model_33.pt"))
    w2v = Wav2Vec(pretrained=True, pretrained_path="../ackit/pretrained/wav2vec_large.pt")
    criterion = nn.CrossEntropyLoss()
    w2v, cl_model, criterion = w2v.to(device=device), cl_model.to(device=device), criterion.to(device)
    configs = {
        "fit": {
            "batch_size": 32,
        },
        "feature": {
            "wav_length": 16000
        }
    }
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    with open("../datasets/d2020_metadata2label.json", 'r', encoding='utf_8') as fp:
        meta2label = json.load(fp)
    train_loader, test_loader = get_type_loader(configs=configs, m2l=meta2label)
    ypred_eval = None
    ytrue_eval = None
    acc_list = []
    cl_model.eval()
    with torch.no_grad():
        for jdx, (wav, mtp, _) in enumerate(tqdm(test_loader, desc="Test")):
            wav = wav.to(device)
            mtp = mtp.to(device)
            mel = w2v(wav)
            pred = cl_model(x=mel)
            # loss_eval = criterion(pred, mtp)
            # print(y_label.shape, pred.shape)
            if jdx == 0:
                ytrue_eval = mtp
                ypred_eval = pred
            else:
                ytrue_eval = torch.concat((ytrue_eval, mtp), dim=0)
                ypred_eval = torch.concat((ypred_eval, pred), dim=0)
            acc_batch = metrics.accuracy_score(mtp.data.cpu().numpy(),
                                               pred.argmax(-1).data.cpu().numpy())
            acc_list.append(acc_batch)
            # print(ytrue_eval.shape, ypred_eval.shape)
    print("accuracy:", np.array(acc_list).mean())
    ytrue_eval = ytrue_eval.data.cpu().numpy()
    ypred_eval = ypred_eval.argmax(-1).data.cpu().numpy()
    print(ytrue_eval.shape, ypred_eval.shape)

    # def get_heat_map(pred_matrix, label_vec, savepath):
    savepath = "../runs/dcase20cls/202406161503w2n7c/result_hm.png"
    max_arg = list(ypred_eval)
    conf_mat = metrics.confusion_matrix(max_arg, ytrue_eval)
    # conf_mat = conf_mat / conf_mat.sum(axis=1)
    ab2full = ["ToyCar", "ToyConveyor", "Fan", "Pump", "Slider", "Valve"]
    df_cm = pd.DataFrame(conf_mat, index=ab2full, columns=ab2full)
    # heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')  # , cbar_kws={'format': '%.2f%'})
    heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')  # , cbar_kws={'format': '%.2f%'})
    # heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues')  # , cbar_kws={'format': '%.2f%'})
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel("Predict Label")
    plt.xlabel("True Label")
    plt.savefig(savepath)
    plt.show()


if __name__ == '__main__':
    train()

    # w2v = Wav2Vec(pretrained=True, pretrained_path="../ackit/pretrained/wav2vec_large.pt")
    # # [20785, 20940]  都是128
    # for i in range(20780, 20960, 5):
    #     print(i, w2v(torch.rand(size=(1, i))).shape)  # 128

    # print(w2v(torch.rand(size=(1, 16000))).shape)

