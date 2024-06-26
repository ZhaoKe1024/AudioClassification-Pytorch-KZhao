#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/5/8 9:44
# @Author: ZhaoKe
# @File : dsptcls.py
# @Software: PyCharm
# Dataset
# Pretrain
# Classifier
import os
import sys
from datetime import datetime
import random
from torch import optim
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import transforms
from ackit.data_utils.transforms import *

from ackit.us8k_cnncls import CNNNet
from ackit.models.mobilenetv2 import MobileNetV2
from ackit.modules.classifiers import LSTM_Classifier, LSTM_Attn_Classifier

from ackit.modules.loss import FocalLoss
from ackit.utils.utils import setup_seed
from ackit.data_utils.us8k import UrbanSound8kDataset


class TrainerSet(object):
    def __init__(self, configs: dict):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.use_pt = configs["pretrained"]
        self.use_data = configs["use_data"]
        self.use_cls = configs["Model"]
        self.configs = configs
        self.batch_size, self.epoch_num = configs["batch_size"], configs["epoch_num"]
        self.train_loader, self.valid_loader = None, None
        self.model = None
        self.save_dir = configs["save_dir"]
        os.makedirs(f"../runs/dsptcls/{self.save_dir}/", exist_ok=True)
        with open(f"../runs/dsptcls/" + self.save_dir + "/info.txt", 'w') as fout:
            fout.write(f"Data: {self.use_data} feature:{configs['features']}\n")
            fout.write(f"Pretrained: {self.use_pt}; Model: {self.use_cls},Loss: {configs['LossFn']}\n")
            fout.write(f"LR: {configs['learning_rate']}, schedule: {configs['lr_scheduler']}\n")
            fout.write(f"epoch_num: {self.epoch_num}, batch_size: {self.batch_size}")

    def __setup_dataset(self):
        if self.use_data == "us8k":
            # use_cls = "cnn"
            self.us8k_df = pd.read_pickle("F:/DATAS/UrbanSound8K/us8k_df.pkl")
        elif self.use_data == "dcase2024":
            # use_cls = "cnn"
            self.us8k_df = pd.read_pickle("F:/DATAS/DCASE2024Task2ASD/us8k_df.pkl")
        elif self.use_data == "coughvid":
            # self.us8k_df = pd.read_pickle("F:/DATAS/COUGHVID-public_dataset_v3/coughvid_df.pkl")
            # self.us8k_df = pd.read_pickle("F:/DATAS/COUGHVID-public_dataset_v3/coughvid_fine_df.pkl")
            self.us8k_df = pd.read_pickle("F:/DATAS/COUGHVID-public_dataset_v3/coughvid_split_specdf.pkl")
        elif self.use_data == "covid19":
            self.us8k_df = pd.read_pickle("F:/DATAS/covid-19-main/dataset-main/covid19_split_balancevalid_df.pkl")

            # self.us8k_df = pd.read_pickle("F:/DATAS/covid-19-main/dataset-main/covid19_split_df.pkl")
        else:
            raise Exception("no data pickle!")
        print(self.us8k_df.head())

        if self.use_data in ["coughvid", "covid19"]:
            self.train_transforms = transforms.Compose([MyRightShift(input_size=(128, 64),
                                                                     width_shift_range=7,
                                                                     shift_probability=0.9),
                                                        MyAddGaussNoise(input_size=(128, 64),
                                                                        add_noise_probability=0.55),
                                                        MyReshape(output_size=(1, 128, 64))])
            self.test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 64))])
        elif self.use_data in ["us8k", "dcase2024"]:
            # build transformation pipelines for data augmentation
            self.train_transforms = transforms.Compose([MyRightShift(input_size=128,
                                                                     width_shift_range=13,
                                                                     shift_probability=0.9),
                                                        MyAddGaussNoise(input_size=128,
                                                                        add_noise_probability=0.55),
                                                        MyReshape(output_size=(1, 128, 128))])
            self.test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 128))])
        else:
            raise Exception("no data pickle!")

    def __get_fold(self, fold_k, batch_size=32):
        # split the data
        if self.use_data == "covid19":
            neg_list = list(range(100)) + list(range(200, 1109))
            pos_list = list(range(100, 200)) + list(range(1109, 2733))
            random.shuffle(neg_list)
            random.shuffle(pos_list)
            valid_list = neg_list[:100] + pos_list[:100]
            train_list = neg_list[100:] + pos_list[100:]
            train_df = self.us8k_df.iloc[train_list, :]
            valid_df = self.us8k_df.iloc[valid_list, :]
        elif self.use_data == "coughvid":
            neg_list = list(range(2076))
            pos_list = list(range(2076, 2850))
            random.shuffle(neg_list)
            random.shuffle(pos_list)
            valid_list = neg_list[:100] + pos_list[:100]
            train_list = neg_list[100:] + pos_list[100:]
            train_df = self.us8k_df.iloc[train_list, :]
            valid_df = self.us8k_df.iloc[valid_list, :]
        else:
            train_df = self.us8k_df[self.us8k_df['fold'] != fold_k]
            valid_df = self.us8k_df[self.us8k_df['fold'] == fold_k]

        # normalize the data
        train_df, valid_df = normalize_data(train_df, valid_df)

        # init train data loader
        train_ds = UrbanSound8kDataset(train_df, transform=self.train_transforms)
        self.train_loader = DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       num_workers=0)

        # init test data loader
        valid_ds = UrbanSound8kDataset(valid_df, transform=self.test_transforms)
        self.valid_loader = DataLoader(valid_ds,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       pin_memory=True,
                                       num_workers=0)

    def __setup_model(self):
        if self.use_cls == "cnn_avgpool":
            self.model = CNNNet(device=self.device).to(self.device)
            # 自带 criterion和optim
        elif self.use_cls == "mnv2":
            self.model = MobileNetV2(dc=1, n_class=2, input_size=64).to(self.device)
        elif self.use_cls == "lstm_vanilla":
            self.model = LSTM_Classifier(inp_size=128, hidden_size=128, n_classes=2).to(self.device)
        elif self.use_cls == "lstm_atten":
            self.model = LSTM_Attn_Classifier(inp_size=128, hidden_size=128, n_classes=2).to(self.device)
        else:
            raise Exception("unknown model!")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.configs["learning_rate"], eps=1e-07,
                                    weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=11,
                                                                    eta_min=1e-4)
        if self.configs["LossFn"] == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.configs["LossFn"] == "FocalLoss":
            self.criterion = FocalLoss(class_num=2).to(self.device)
        else:
            raise Exception("unknown LossFn!")

    def test_run(self):
        setup_seed(seed=3407)
        self.__setup_dataset()
        self.__setup_model()
        self.__get_fold(fold_k=0, batch_size=self.batch_size)
        self.__train_epoch(0)

    def __train_epoch(self, epoch_id):
        print("\nEpoch {}/{}".format(epoch_id + 1, self.epoch_num))
        with tqdm(total=len(self.train_loader), file=sys.stdout) as pbar:
            for step, batch in enumerate(self.train_loader):
                X_batch = batch['spectrogram'].to(torch.float32).to(self.device)
                y_batch = batch['label'].to(self.device)
                # print(X_batch.shape, y_batch.shape)
                # return
                # zero the parameter gradients
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward + backward
                    outputs = self.model(X_batch)
                    batch_loss = self.criterion(outputs, y_batch)
                    batch_loss.backward()
                    # update the parameters
                    self.optimizer.step()
                pbar.update(1)

                # model evaluation - train data
        train_loss, train_acc = self.__evaluate(dataloader=self.train_loader)
        print("loss: %.4f - accuracy: %.4f" % (train_loss, train_acc), end='')
        return train_loss, train_acc

    def __evaluate(self, dataloader):
        running_loss = torch.tensor(0.0).to(self.device)
        running_acc = torch.tensor(0.0).to(self.device)

        batch_size = torch.tensor(dataloader.batch_size).to(self.device)
        # model evaluation - validation data
        for step, batch in enumerate(dataloader):
            X_batch = batch['spectrogram'].to(torch.float32).to(self.device)
            y_batch = batch['label'].to(self.device)

            # outputs = model.predict(X_batch)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_batch)

            # get batch loss
            loss = self.criterion(outputs, y_batch)
            running_loss = running_loss + loss

            # calculate batch accuracy
            # print(y_batch)
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions = (predictions == y_batch).float().sum()
            running_acc = running_acc + torch.div(correct_predictions, batch_size)

        loss = running_loss.item() / (step + 1)
        accuracy = running_acc.item() / (step + 1)

        return loss, accuracy

    def train(self):
        setup_seed(seed=3407)
        self.__setup_dataset()
        self.__setup_model()
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        # self.__get_fold(fold_k=0)
        # score = self.__evaluate(dataloader=self.valid_loader)
        start_time = datetime.now()
        for epoch_id in range(self.epoch_num):
            self.__get_fold(fold_k=epoch_id % 10 + 1)  # covid19 random select

            # train the model
            # print("Pre-training accuracy: %.4f%%" % (100 * score[1]))
            self.model.train()
            train_loss, train_acc = self.__train_epoch(epoch_id=epoch_id)

            # model evaluation - validation data
            val_loss, val_acc = None, None
            if self.valid_loader is not None:
                val_loss, val_acc = self.__evaluate(dataloader=self.valid_loader)
                print(" - val_loss: %.4f - val_accuracy: %.4f" % (val_loss, val_acc))

            # store the model's training progress
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            if epoch_id % 10 == 9:
                self.show_results(history, epoch_id)

            if epoch_id > 100 and epoch_id % 10 == 9:
                # os.makedirs("../runs/dsptcls/covid19fl/", exist_ok=True)
                torch.save(self.model.state_dict(), f"../runs/dsptcls/{self.save_dir}/ckpt_epoch{epoch_id}.pt")
            if epoch_id >= self.configs["start_scheduler_epoch"]:
                self.scheduler.step()
        end_time = datetime.now() - start_time
        print("\nTraining completed in time: {}".format(end_time))

    # def show_results(self, tot_history, name):
    def show_results(self, history, name):
        """Show accuracy and loss graphs for train and test sets."""

        # for i, history in enumerate(tot_history):
        # print('\n({})'.format(i + 1))
        print('\n({})'.format(name))

        plt.figure(figsize=(15, 5))

        plt.subplot(121)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.grid(linestyle='--')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(122)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.grid(linestyle='--')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        os.makedirs(f"../runs/dsptcls/{self.save_dir.split('/')[0]}/", exist_ok=True)
        plt.savefig(f"../runs/dsptcls/{self.save_dir}/validloss_{name}.png", format="png", dpi=300)
        plt.close()
        # plt.show()

        print('\tMax validation accuracy: %.4f %%' % (np.max(history['val_accuracy']) * 100))
        print('\tMin validation loss: %.5f' % np.min(history['val_loss']))

    def load_ckpt(self, resume_model_path):
        if self.model is None:
            self.__setup_model()
        state_dict = torch.load(resume_model_path)
        self.model.load_state_dict(state_dict)

    def test(self, resume_model_path):
        self.load_ckpt(resume_model_path=resume_model_path)
        self.__setup_dataset()
        self.__get_fold(fold_k=0)

        running_loss = torch.tensor(0.0).to(self.device)
        running_acc = torch.tensor(0.0).to(self.device)

        self.model.eval()

        batch_size = torch.tensor(self.train_loader.batch_size).to(self.device)
        # model evaluation - validation data
        for step, batch in enumerate(self.train_loader):
            X_batch = batch['spectrogram'].to(torch.float32).to(self.device)
            y_batch = batch['label']

            # outputs = model.predict(X_batch)
            with torch.no_grad():
                outputs = self.model(X_batch)

            # # get batch loss
            # loss = self.model.criterion(outputs, y_batch)
            # running_loss = running_loss + loss

            # calculate batch accuracy
            # print(y_batch)
            predictions = torch.argmax(outputs, dim=1)

            predictions = predictions.data.cpu().numpy()
            y_batch = y_batch.data.cpu().numpy()
            # print(predictions)
            # print(y_batch)

            precision = metrics.precision_score(y_batch, predictions)
            recall = metrics.recall_score(y_batch, predictions)
            correct = metrics.accuracy_score(y_batch, predictions)
            print(f"precision: {precision:.5f}, recall: {recall:.5f}, accuracy: {correct:.5f}")

            conf_mat = metrics.confusion_matrix(predictions, y_batch)
            print(conf_mat)

        # loss = running_loss.item() / (step + 1)
        # accuracy = running_acc.item() / (step + 1)


if __name__ == '__main__':
    # train
    # Run paras
    data_list = ["us8k", "coughvid", "covid19"]
    loss_list = ["FocalLoss", "CrossEntropyLoss"]
    model_list = ["cnn_avgpool", "mnv2", "lstm_vanilla", "lstm_atten"]
    run_config = {
        "use_data": data_list[1],
        "input_shape": (128, 64),
        "pretrained": None,
        "Model": model_list[1],
        "LossFn": loss_list[0],
        "features": "Melspec",
        "learning_rate": 2e-4,
        "lr_scheduler": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "start_scheduler_epoch": 11,
        "epoch_num": 220,
        "batch_size": 128,
        "save_dir": "coughvid202406111730lstm",
    }
    trainer = TrainerSet(run_config)
    trainer.train()
    # # trainer.test_run()

    # trainer.load_ckpt(resume_model_path="../runs/dsptcls/coughvid202405181711mnv2_finecycle/ckpt_epoch219.pt")
    # trainer.test(resume_model_path="../runs/dsptcls/coughvid202405181711mnv2_finecycle/ckpt_epoch219.pt")

    # trainer = TrainerSet(use_data="covid19", use_cls="cnn_avgpool")
    # trainer.train()

    # test
    # trainer = TrainerSet(use_data="covid19", use_cls="cnn_avgpool")
    # # trainer.load_ckpt(resume_model_path="../runs/dsptcls/covid19_notcross/covid19_split_149.pt")
    # trainer.test(resume_model_path="../runs/dsptcls/covid19_notcross/covid19_split_149.pt")

    # x = np.random.randn(int(2.95*22050))
    # # x = np.random.randn(int(2.95*16000))
    # print(x.shape)
    # import librosa
    # melspectrogram = librosa.feature.melspectrogram(y=x,
    #                                                 sr=22050,
    #                                                 hop_length=512,
    #                                                 win_length=512,
    #                                                 n_mels=128)
    # print(melspectrogram.shape)
