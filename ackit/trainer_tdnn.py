#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/22 16:38
# @Author: ZhaoKe
# @File : trainer_tdnn.py
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

from ackit.trainer_setting import get_model
from ackit.utils.utils import weight_init, load_ckpt
from ackit.data_utils.sound_reader import get_former_loader


class TrainerTDNN():
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
                                    "run_save_dir"] + "tdnn/" + self.timestr + f'_tdnn/'
            if istrain:
                os.makedirs(self.run_save_dir, exist_ok=True)
        with open("./datasets/d2020_metadata2label.json", 'r', encoding='utf_8') as fp:
            self.meta2label = json.load(fp)

    def train(self):
        model = get_model("tdnn", self.configs, istrain=True).to(self.device)
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
                # print("shape of x_mel:", x_mel.shape)
                x_mel = x_mel / 255.
                optimizer.zero_grad()
                mtid = torch.tensor(mtid, device=self.device)
                mtid_pred = model(x=x_mel)
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
            if epoch > 6 and epoch % 2 == 0:
                os.makedirs(self.run_save_dir + f"model_epoch_{epoch}/", exist_ok=True)
                tmp_model_path = "{model}model_{epoch}.pth".format(
                    model=self.run_save_dir + f"model_epoch_{epoch}/",
                    epoch=epoch)
                torch.save(model.state_dict(), tmp_model_path)

            if epoch >= self.configs["model"]["start_scheduler_epoch"]:
                scheduler.step()
        print("============== END TRAINING ==============")

    def test_tsne(self, resume_model_path, load_epoch):
        model = get_model("tdnn", self.configs, istrain=False).to(self.device)
        if load_epoch is not None:
            load_ckpt(model, resume_model_path, load_epoch=load_epoch)
        else:
            load_ckpt(model, resume_model_path)
        model.eval()
        print("---------------train dataset-------------")
        train_loader, _ = get_former_loader(istrain=True, istest=False, configs=self.configs,
                                            meta2label=self.meta2label, isdemo=True)

        with torch.no_grad():
            metrics_input = None
            # mtypes = None
            mtids = None
            for id_idx, (x_mel, mtype, mtid) in enumerate(tqdm(train_loader, desc="Testing")):
                x_mel = x_mel / 255.
                pred = model(x=x_mel)
                if id_idx == 0:
                    metrics_input = pred
                    # mtypes = mtype
                    mtids = mtid
                else:
                    metrics_input = torch.concatenate([metrics_input, pred], dim=0)
                    # mtypes = torch.concatenate([mtids, mtype], dim=0)
                    mtids = torch.concatenate([mtids, mtid], dim=0)
                    # labels = torch.concatenate([labels, y_true], dim=0)

            metrics_input = metrics_input.data.cpu().numpy()
            # mtypes = mtypes.data.cpu().numpy()
            mtids = mtids.data.cpu().numpy()
            # labels = labels.data.cpu().numpy()
            print("tnse shape:", metrics_input.shape)
            # print("type shape:", mtypes.shape)
            print("mtid shape:", mtids.shape)
            from ackit.utils.metrics import get_heat_map
            get_heat_map(pred_matrix=metrics_input,
                         label_vec=mtids,
                         savepath=resume_model_path + f'/heatmap_epoch{load_epoch}.png')
            # from sklearn.manifold import TSNE
            # from asdkit.utils.plotting import plot_embedding_2D
            # tsne_model = TSNE(n_components=2, init="pca", random_state=3407)
            # result2D = tsne_model.fit_transform(metrics_input)
            # print("TSNE finish.")
            # plot_embedding_2D(result2D, mtids, "t-SNT for type and id",
            #                   resume_model_path + f'/TSNE_mtid_epoch{load_epoch}.png')
