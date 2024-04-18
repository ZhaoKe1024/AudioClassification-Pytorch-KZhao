#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/16 19:09
# @Author: ZhaoKe
# @File : trainer_ConvEncoder.py
# @Software: PyCharm
"""

"""
import os
import yaml
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from ackit.trainer_setting import get_model
from ackit.utils.utils import load_ckpt, setup_seed
from ackit.data_utils.soundreader2020 import get_former_loader


class TrainerEncoder(object):
    def __init__(self, configs="../configs/autoencoder.yaml", istrain=True, isdemo=True):
        self.configs = None
        with open(configs) as stream:
            self.configs = yaml.safe_load(stream)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.num_epoch = self.configs["fit"]["epochs"]
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.demo_test = not istrain
        self.run_save_dir = self.configs[
                                "run_save_dir"] + '/'
        if istrain:
            self.run_save_dir += self.timestr + f'_ptvae_lr-3/'
            if not isdemo:
                os.makedirs(self.run_save_dir, exist_ok=True)

        with open("../datasets/d2020_metadata2label.json", 'r', encoding='utf_8') as fp:
            self.meta2label = json.load(fp)
        self.pretrain_model = None
        self.model = None
        self.trainloader = None

    def __setup_datasets(self, evaluate=False):
        if evaluate:
            self.trainloader, _ = get_former_loader(istrain=True, istest=False,
                                                    meta2label=self.meta2label, configs=self.configs, isdemo=True)
        else:
            self.trainloader, _ = get_former_loader(istrain=True, istest=False,
                                                    meta2label=self.meta2label, configs=self.configs, isdemo=False)

    def __setup_models(self, pretrain=True):
        self.model = get_model("cnn_classifier", self.configs, istrain=True)
        if pretrain:
            self.__load_pretrain(resume_path="../runs/VAE/model_epoch_12")
            self.pretrain_model.to(self.device)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epoch, eta_min=5e-5)
        self.class_loss = nn.CrossEntropyLoss().to(self.device)
        print("All model and loss are on device:", self.device)

    def train_classifier(self):
        setup_seed(3407)
        self.__setup_datasets(evaluate=False)
        self.__setup_models(pretrain=True)

        history1 = []
        for epoch in range(self.num_epoch):
            self.model.train()
            for x_idx, (x_mel, mtype, _) in enumerate(tqdm(self.trainloader, desc="Training")):
                x_mel = x_mel.unsqueeze(1) / 255.
                x_mel = x_mel.to(self.device)
                mtype = torch.tensor(mtype, device=self.device)
                # print(x_mel.shape, mtype.shape)
                with torch.no_grad():
                    featmap = self.pretrain_model(x_mel, featmap_only=True)
                # print(x_mel.shape, mtype.shape)
                self.optimizer.zero_grad()
                mtid_pred, _ = self.model(featmap)
                pred_loss = self.class_loss(mtid_pred, mtype)

                pred_loss.backward()
                self.optimizer.step()

                history1.append(pred_loss.item())
                if x_idx % 60 == 0:
                    print(f"Epoch[{epoch}], mtid pred loss:{pred_loss.item():.4f}")

            plt.figure()
            plt.plot(range(len(history1)), history1, c="green", alpha=0.7)
            plt.savefig(self.run_save_dir + f'mtid_loss_iter_{epoch}.png')
            plt.close()

            os.makedirs(self.run_save_dir + f"model_epoch_{epoch}/", exist_ok=True)
            tmp_model_path = "{model}model_{epoch}.pth".format(
                model=self.run_save_dir + f"model_epoch_{epoch}/",
                epoch=epoch)
            torch.save(self.model.state_dict(), tmp_model_path)

            if epoch >= self.configs["model"]["start_scheduler_epoch"]:
                self.scheduler.step()
        print("============== END TRAINING ==============")

    def __load_pretrain(self, resume_path):
        self.pretrain_model = get_model(use_model="vae", configs=self.configs, istrain=False).to(self.device)
        cvae_model_path = resume_path
        load_epoch = 12
        load_ckpt(self.pretrain_model, cvae_model_path, load_epoch=load_epoch)

    def __setup_evaluate(self, resume_path="202404181142_ptvae", load_epoch=119):
        setup_seed(3407)
        if self.trainloader is None:
            self.__setup_datasets(evaluate=True)
        if self.pretrain_model is None:
            self.__load_pretrain(resume_path="../runs/VAE/model_epoch_12")
            self.pretrain_model.to(self.device)
            self.pretrain_model.eval()
        if self.model is None:
            self.model = get_model("cnn_classifier", configs=self.configs, istrain=False).to(self.device)
            load_ckpt(self.model, self.run_save_dir + resume_path + f"/model_epoch_{load_epoch}", load_epoch=load_epoch)
            self.model.eval()

    def plot_reduction(self, resume_path="202404181142_ptvae", load_epoch=119):
        self.__setup_evaluate(resume_path, load_epoch)
        tsne_input = None
        labels = None
        for x_idx, (x_mel, mtype, _) in enumerate(tqdm(self.trainloader, desc="Training")):
            x_mel = x_mel.unsqueeze(1) / 255.
            x_mel.to(self.device)
            mtype = torch.tensor(mtype, device=self.device)
            # print(x_mel.shape, mtype.shape)
            with torch.no_grad():
                featmap = self.pretrain_model(x_mel, featmap_only=True)

            _, fm = self.model(featmap)
            if x_idx == 0:
                tsne_input, labels = fm, mtype
            else:
                tsne_input = torch.concat((tsne_input, fm), dim=0)
                labels = torch.concat((labels, mtype), dim=0)
        print("tsne_input shape:", tsne_input.shape)
        print("lables shape:", labels.shape)
        from ackit.utils.plotter import plot_tSNE
        plot_tSNE(embd=tsne_input.detach().cpu().numpy(), names=labels.detach().cpu().numpy(), save_path=self.run_save_dir + resume_path + f"/tsne_epoch_79.png")

    def plot_heatmap(self, resume_path="202404181142_ptvae", load_epoch=119):
        self.__setup_evaluate(resume_path=resume_path, load_epoch=load_epoch)
        heatmap_input = None
        labels = None
        for x_idx, (x_mel, mtype, _) in enumerate(tqdm(self.trainloader, desc="Training")):
            x_mel = x_mel.unsqueeze(1) / 255.
            x_mel.to(self.device)
            mtype = torch.tensor(mtype, device=self.device)
            # print(x_mel.shape, mtype.shape)
            with torch.no_grad():
                featmap = self.pretrain_model(x_mel, featmap_only=True)

            pred, _ = self.model(featmap)
            if x_idx == 0:
                heatmap_input, labels = pred, mtype
            else:
                heatmap_input = torch.concat((heatmap_input, pred), dim=0)
                labels = torch.concat((labels, mtype), dim=0)
        print("heatmap_input shape:", heatmap_input.shape)
        print("lables shape:", labels.shape)
        from ackit.utils.plotter import calc_accuracy, plot_heatmap
        heatmap_input = heatmap_input.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        calc_accuracy(pred_matrix=heatmap_input, label_vec=labels, save_path=self.run_save_dir + resume_path + f"/accuracy_epoch_{load_epoch}.txt")
        plot_heatmap(pred_matrix=heatmap_input, label_vec=labels, save_path=self.run_save_dir + resume_path + f"/heatmap_epoch_{load_epoch}.png")


if __name__ == '__main__':
    # trainer = TrainerEncoder(istrain=True, isdemo=False)
    # trainer.train_classifier()

    trainer = TrainerEncoder(istrain=False, isdemo=False)
    # trainer.plot_reduction(resume_path="202404181446_ptvae_lr-3", load_epoch=119)
    trainer.plot_heatmap(resume_path="202404181446_ptvae_lr-3", load_epoch=119)
