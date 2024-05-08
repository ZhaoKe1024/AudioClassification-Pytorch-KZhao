#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/5/8 9:44
# @Author: ZhaoKe
# @File : dsptcls.py
# @Software: PyCharm
# Dataset
# Pretrain
# Classifier
import sys
from datetime import datetime
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from ackit.us8k_cnncls import CNNNet
from ackit.utils.utils import setup_seed
from ackit.data_utils.transforms import *
from ackit.data_utils.us8k import UrbanSound8kDataset


class TrainerSet(object):
    def __init__(self, use_data="coughvid", use_pt=None, use_cls="cnn"):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.use_pt = use_pt
        self.use_data = use_data
        self.use_cls = use_cls
        self.cls = None
        self.epoch_num = 150
        self.train_loader, self.valid_loader = None, None

    def __setup_dataset(self):
        if self.use_data == "us8k":
            self.us8k_df = pd.read_pickle("F:/DATAS/UrbanSound8K/us8k_df.pkl")

        elif self.use_data == "dcase2024":
            self.us8k_df = pd.read_pickle("F:/DATAS/DCASE2024Task2ASD/us8k_df.pkl")
        elif self.use_data == "coughvid":
            self.us8k_df = pd.read_pickle("F:/DATAS/COUGHVID-public_dataset_v3/coughvid_df.pkl")
        else:
            raise Exception("no data pickle!")
        print(self.us8k_df.head())
        # build transformation pipelines for data augmentation
        self.train_transforms = transforms.Compose([MyRightShift(input_size=128,
                                                                 width_shift_range=13,
                                                                 shift_probability=0.9),
                                                    MyAddGaussNoise(input_size=128,
                                                                    add_noise_probability=0.55),
                                                    MyReshape(output_size=(1, 128, 128))])

        self.test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 128))])

    def __get_fold(self, fold_k, batch_size=32):
        # split the data
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
        if self.use_cls == "cnn":
            self.model = CNNNet(device=self.device).to(self.device)
            # 自带 criterion和optim

    def __train_epoch(self, epoch_id):
        print("\nEpoch {}/{}".format(epoch_id + 1, self.epoch_num))
        with tqdm(total=len(self.train_loader), file=sys.stdout) as pbar:
            for step, batch in enumerate(self.train_loader):
                X_batch = batch['spectrogram'].to(torch.float32).to(self.device)
                y_batch = batch['label'].to(self.device)

                # zero the parameter gradients
                self.model.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    # forward + backward
                    outputs = self.model(X_batch)
                    batch_loss = self.model.criterion(outputs, y_batch)
                    batch_loss.backward()
                    # update the parameters
                    self.model.optimizer.step()
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
            loss = self.model.criterion(outputs, y_batch)
            running_loss = running_loss + loss

            # calculate batch accuracy
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

        # score = self.__evaluate(dataloader=self.valid_loader)
        start_time = datetime.now()
        for epoch_id in range(self.epoch_num):
            self.__get_fold(fold_k=epoch_id % 10 + 1)

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
            if epoch_id % 10 == 0:
                self.show_results(history, epoch_id)
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
        plt.savefig(f"../runs/dsptcls/coughvid_{name}.png", format="png", dpi=300)
        # plt.show()

        print('\tMax validation accuracy: %.4f %%' % (np.max(history['val_accuracy']) * 100))
        print('\tMin validation loss: %.5f' % np.min(history['val_loss']))


if __name__ == '__main__':
    trainer = TrainerSet()
    trainer.train()

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
