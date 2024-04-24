#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/22 16:38
# @Author: ZhaoKe
# @File : trainer_tdnn.py
# @Software: PyCharm
import os
import yaml
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ackit.data_utils.collate_fn import collate_fn
from ackit.trainer_setting import get_model
from ackit.utils.utils import load_ckpt
from ackit.data_utils.coughvid_reader import CoughVID_Dataset
from ackit.data_utils.featurizer import Wave2Mel
from ackit.utils.plotter import calc_accuracy, plot_heatmap


class TrainerTDNN(object):
    def __init__(self, configs="../configs/tdnn.yaml", istrain=True, isdemo=True):
        self.configs = None
        with open(configs) as stream:
            self.configs = yaml.safe_load(stream)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.num_epoch = self.configs["fit"]["epochs"]
        self.timestr = time.strftime("%Y%m%d%H%M", time.localtime())
        self.demo_test = not istrain
        self.isdemo = isdemo
        if istrain:
            self.run_save_dir = self.configs[
                                    "run_save_dir"] + self.timestr + f'_tdnn/'
            if not isdemo:
                os.makedirs(self.run_save_dir, exist_ok=True)

    def __setup_dataloader(self):
        self.train_dataset = CoughVID_Dataset(root_path="./datasets/waveinfo_annotation.csv", configs=self.configs, isdemo=self.isdemo)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.configs["fit"]["batch_size"], shuffle=True,
                                       collate_fn=collate_fn)
        self.w2m = Wave2Mel(sr=16000, n_mels=80)

    def __setup_model(self):
        self.model = get_model("tdnn", self.configs, istrain=True).to(self.device)
        self.cls_loss = nn.CrossEntropyLoss().to(self.device)
        print("All model and loss are on device:", self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=5e-5)

    def train(self):
        self.__setup_dataloader()
        self.__setup_model()
        history1 = []
        for epoch_id in range(self.num_epoch):
            # ---------------------------
            # -----------TRAIN-----------
            # ---------------------------
            self.model.train()
            for x_idx, (x_wav, y_label, _) in enumerate(tqdm(self.train_loader, desc="Training")):
                x_mel = self.w2m(x_wav).to(self.device)
                y_label = torch.tensor(y_label, device=self.device)
                # print("shape of x_mel:", x_mel.shape)
                self.optimizer.zero_grad()
                y_pred = self.model(x=x_mel)
                # recon_loss = self.recon_loss(recon_spec, x_mel)
                pred_loss = self.cls_loss(y_pred, y_label)
                pred_loss.backward()
                self.optimizer.step()

                if x_idx > 2:
                    history1.append(pred_loss.item())
                if x_idx % 60 == 0:
                    print(f"Epoch[{epoch_id}], mtid pred loss:{pred_loss.item():.4f}")
            if epoch_id >= self.configs["model"]["start_scheduler_epoch"]:
                self.scheduler.step()

            # ---------------------------
            # -----------SAVE------------
            # ---------------------------
            plt.figure(0)
            plt.plot(range(len(history1)), history1, c="green", alpha=0.7)
            plt.savefig(self.run_save_dir + f'cls_loss_iter_{epoch_id}.png')
            # if epoch > 6 and epoch % 2 == 0:
            os.makedirs(self.run_save_dir + f"model_epoch_{epoch_id}/", exist_ok=True)
            tmp_model_path = "{model}model_{epoch}.pth".format(
                model=self.run_save_dir + f"model_epoch_{epoch_id}/",
                epoch=epoch_id)
            torch.save(self.model.state_dict(), tmp_model_path)
            # ---------------------------
            # -----------TEST------------
            # ---------------------------
            self.model.eval()
            heatmap_input = None
            labels = None
            for x_idx, (x_wav, y_label, _) in enumerate(tqdm(self.train_loader, desc="Test")):
                x_mel = self.w2m(x_wav).to(self.device)
                y_label = torch.tensor(y_label, device=self.device)
                y_pred = self.model(x=x_mel)
                if x_idx == 0:
                    heatmap_input, labels = y_pred, y_label
                else:
                    heatmap_input = torch.concat((heatmap_input, y_pred), dim=0)
                    labels = torch.concat((labels, y_label), dim=0)
                if x_idx * self.configs["fit"]["batch_size"] > 800:
                    break
            print("heatmap_input shape:", heatmap_input.shape)
            print("lables shape:", labels.shape)
            # if epoch > 3:
            #     self.plot_reduction(resume_path="", load_epoch=epoch, reducers=["heatmap"])
            heatmap_input = heatmap_input.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            calc_accuracy(pred_matrix=heatmap_input, label_vec=labels, save_path=None)
            plot_heatmap(pred_matrix=heatmap_input, label_vec=labels,
                         ticks=["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"],
                         save_path=self.run_save_dir + f"/heatmap_epoch_{epoch_id}.png")
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
