#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/16 19:09
# @Author: ZhaoKe
# @File : trainer_ConvEncoder.py
# @Software: PyCharm
import os
import yaml
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from ackit.utils.utils import weight_init
from ackit.data_utils.featurizer import Wave2Mel
from ackit.models.autoencoder import ConvEncoder
from ackit.data_utils.sound_reader import get_former_loader


class TrainerEncoder(object):
    def __init__(self, configs="../configs/conformer.yaml", istrain=True):
        self.configs = None
        with open(configs) as stream:
            self.configs = yaml.safe_load(stream)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.num_epoch = self.configs["fit"]["epochs"]
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.demo_test = not istrain
        if istrain:
            self.run_save_dir = self.configs[
                                    "run_save_dir"] + self.timestr + f'_v3tfs_w2m/'
            if istrain:
                os.makedirs(self.run_save_dir, exist_ok=True)
        self.w2m = Wave2Mel(16000)

        with open("./datasets/metadata2label.json", 'r', encoding='utf_8') as fp:
            self.meta2label = json.load(fp)
        self.id2map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
        self.mt2id = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}

    def train_encoder(self):
        model = ConvEncoder(input_channel=1, input_length=self.configs["model"]["input_length"],
                            input_dim=self.configs["feature"]["n_mels"],
                            class_num=self.configs["model"]["mtid_class_num"],
                            class_num1=self.configs["model"]["type_class_num"]).to(self.device)
        model.apply(weight_init)
        class_loss = nn.CrossEntropyLoss().to(self.device)
        print("All model and loss are on device:", self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=5e-5)

        train_loader, _ = get_former_loader(istrain=True, istest=False, configs=self.configs,
                                            meta2label=self.meta2label,
                                            isdemo=self.demo_test)
        history1 = []
        for epoch in range(self.num_epoch):
            model.train()
            for x_idx, (x_mel, mtype, mtid) in enumerate(tqdm(train_loader, desc="Training")):
                x_mel = x_mel.unsqueeze(1) / 255.
                optimizer.zero_grad()
                mtid = torch.tensor(mtid, device=self.device)
                feat, mtid_pred = model(input_mel=x_mel, class_vec=mtid, coarse_cls=False, fine_cls=True)
                # recon_loss = self.recon_loss(recon_spec, x_mel)
                mtid_pred_loss = class_loss(mtid_pred, mtid)
                mtid_pred_loss.backward()
                optimizer.step()
                history1.append(mtid_pred_loss.item())
                if x_idx % 60 == 0:
                    print(f"Epoch[{epoch}], mtid pred loss:{mtid_pred_loss.item():.4f}")
            if epoch % 1 == 0:
                plt.figure(2)
                plt.plot(range(len(history1)), history1, c="green", alpha=0.7)
                plt.savefig(self.run_save_dir + f'mtid_loss_iter_{epoch}.png')
            if epoch > 34 and epoch % 4 == 0:
                os.makedirs(self.run_save_dir + f"model_epoch_{epoch}/", exist_ok=True)
                tmp_model_path = "{model}model_{epoch}.pth".format(
                    model=self.run_save_dir + f"model_epoch_{epoch}/",
                    epoch=epoch)
                torch.save(model.state_dict(), tmp_model_path)

            if epoch >= self.configs["model"]["start_scheduler_epoch"]:
                scheduler.step()
        print("============== END TRAINING ==============")
