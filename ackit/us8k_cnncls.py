#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/5/6 20:18
# @Author: ZhaoKe
# @File : us8k_cnncls.py
# @Software: PyCharm
import os
import sys
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from ackit.data_utils.transforms import *
from ackit.data_utils.us8k import UrbanSound8kDataset


class CNNNet(nn.Module):
    def __init__(self, device):
        super(CNNNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, padding=0)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, padding=0)

        self.fc1 = nn.Linear(in_features=48, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-07, weight_decay=1e-3)

        self.device = device

    def forward(self, x):
        # cnn layer-1
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=3)
        x = F.relu(x)

        # cnn layer-2
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        x = F.relu(x)

        # cnn layer-3
        x = self.conv3(x)
        x = F.relu(x)

        # global average pooling 2D
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(-1, 48)

        # dense layer-1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        # dense output layer
        x = self.fc2(x)

        return x


class US8KCNNTrainer(object):
    def __init__(self):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None

    def __setup_data(self):
        self.us8k_df = pd.read_pickle("F:/DATAS/UrbanSound8K/us8k_df.pkl")
        print(self.us8k_df.head())

        # build transformation pipelines for data augmentation
        self.train_transforms = transforms.Compose([MyRightShift(input_size=128,
                                                                 width_shift_range=13,
                                                                 shift_probability=0.9),
                                                    MyAddGaussNoise(input_size=128,
                                                                    add_noise_probability=0.55),
                                                    MyReshape(output_size=(1, 128, 128))])

        self.test_transforms = transforms.Compose([MyReshape(output_size=(1, 128, 128))])

    def __setup_model(self):
        self.model = CNNNet(self.device).to(self.device)

    def train_epoch(self, fold_k, dataset_df, epochs=100, batch_size=32, num_of_workers=0):
        # split the data
        train_df = dataset_df[dataset_df['fold'] != fold_k]
        test_df = dataset_df[dataset_df['fold'] == fold_k]

        # normalize the data
        train_df, test_df = normalize_data(train_df, test_df)

        # init train data loader
        train_ds = UrbanSound8kDataset(train_df, transform=self.train_transforms)
        train_loader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=num_of_workers)

        # init test data loader
        test_ds = UrbanSound8kDataset(test_df, transform=self.test_transforms)
        test_loader = DataLoader(test_ds,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_of_workers)

        # init model
        self.__setup_model()

        # pre-training accuracy
        score = self.evaluate(self.model, test_loader)
        print("Pre-training accuracy: %.4f%%" % (100 * score[1]))

        # train the model
        start_time = datetime.now()

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        for epoch in range(epochs):
            self.model.train()
            print("\nEpoch {}/{}".format(epoch + 1, epochs))
            with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
                for step, batch in enumerate(train_loader):
                    X_batch = batch['spectrogram'].to(self.device)
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
            train_loss, train_acc = self.evaluate(self.model, train_loader)
            print("loss: %.4f - accuracy: %.4f" % (train_loss, train_acc), end='')

            # model evaluation - validation data
            val_loss, val_acc = None, None
            if test_loader is not None:
                val_loss, val_acc = self.evaluate(self.model, test_loader)
                print(" - val_loss: %.4f - val_accuracy: %.4f" % (val_loss, val_acc))

            # store the model's training progress
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

        end_time = datetime.now() - start_time
        print("\nTraining completed in time: {}".format(end_time))
        return history

    def evaluate(self, model, data_loader):
        running_loss = torch.tensor(0.0).to(self.device)
        running_acc = torch.tensor(0.0).to(self.device)

        batch_size = torch.tensor(data_loader.batch_size).to(self.device)

        for step, batch in enumerate(data_loader):
            X_batch = batch['spectrogram'].to(self.device)
            y_batch = batch['label'].to(self.device)

            # outputs = model.predict(X_batch)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_batch)

            # get batch loss
            loss = model.criterion(outputs, y_batch)
            running_loss = running_loss + loss

            # calculate batch accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions = (predictions == y_batch).float().sum()
            running_acc = running_acc + torch.div(correct_predictions, batch_size)

        loss = running_loss.item() / (step + 1)
        accuracy = running_acc.item() / (step + 1)

        return loss, accuracy

    def train(self):
        for FOLD_K in range(1, 11):

            REPEAT = 3
            self.__setup_data()
            history1 = []

            for i in range(REPEAT):
                print('-' * 80)
                print("\n({})\n".format(i + 1))

                history = self.train_epoch(FOLD_K, self.us8k_df, epochs=100, num_of_workers=4)
                history1.append(history)
            self.show_results(history1, FOLD_K)

    def show_results(self, tot_history, name):
        """Show accuracy and loss graphs for train and test sets."""

        for i, history in enumerate(tot_history):
            print('\n({})'.format(i + 1))

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
            plt.savefig(f"./us8k_res_{name}.png", format="png", dpi=300)
            plt.show()

            print('\tMax validation accuracy: %.4f %%' % (np.max(history['val_accuracy']) * 100))
            print('\tMin validation loss: %.5f' % np.min(history['val_loss']))


if __name__ == '__main__':
    trainer = US8KCNNTrainer()
    trainer.train()
