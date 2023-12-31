#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/22 16:38
# @Author: ZhaoKe
# @File : trainer_ast.py
# @Software: PyCharm
from abc import ABC

import yaml

from ackit.models.ASTransformer import AST
from ackit.models.CNNBaseline import Net
import torch
import os
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, precision_recall_curve

from ackit.utils.utils import dict_to_object


class ASTTrainer():
    def __init__(self, config="../configs/astransformer.yaml"):
        super(ASTTrainer, self).__init__()
        self.configs = None
        if isinstance(config, str):
            with open(config, 'r') as jsf:
                cfg = yaml.load(jsf.read(), Loader=yaml.FullLoader)
                self.configs = dict_to_object(cfg)

    def __setup_dataloader(self, x_train, y_train, x_val=None, y_val=None, shuffle=True, n_channels=1):
        x_train = torch.tensor(x_train).float()
        if n_channels == 3:
            x_train = x_train.repeat(1, 3, 1, 1)  # If using 3 channels e.g. vision transformers
        y_train = torch.tensor(y_train).to(torch.int64)
        y_train = torch.nn.functional.one_hot(y_train, num_classes=self.configs.n_classes)
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=shuffle)

        if x_val is not None:
            x_val = torch.tensor(x_val).float()
            if n_channels == 3:
                x_val = x_val.repeat(1, 3, 1, 1)
            y_val = torch.tensor(y_val).to(torch.int64)
            y_val = torch.nn.functional.one_hot(y_val, num_classes=self.config.n_classes)
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=shuffle)

            return train_loader, val_loader
        return train_loader

    def _setup_model(self):
        pass

    def _train_epoch(self):
        pass

    def train(self):
        cv_fold=2
        use_val=False

        if not use_val:
            x_val = None  # Start with no validation data, but to be improved!
        X_train_all = []
        y_train_all = []
        for j, fold in enumerate([1, 2, 3, 4, 5]):  # Load all remaining data except current fold to be used as training
            if fold != cv_fold:
                pickle_name = ('lms_cv_fold_' + str(fold) + '_len_fft_' + str(self.configs.len_fft) + '_win_len_' + str(
                    self.configs.win_len)
                               + '_hop_len_' + str(self.configs.hop_len) + '_n_mel_' + str(self.configs.n_mel) + '.pkl')

                with open(os.path.join(self.configs.dir_out_feat, pickle_name), 'rb') as f:
                    feat = pickle.load(f)
                    print('Loaded train features from:', os.path.join(self.configs.dir_out_feat, pickle_name))

                    X_train_all.append(feat['X_train'])
                    y_train_all.append(feat['y_train'])

        # Convert to Torch format:
        x_train = np.vstack(X_train_all)
        y_train = np.hstack(y_train_all)

        print('X_train, y_train', np.shape(x_train), np.shape(y_train))

        input_tdim = np.shape(x_train)[1]  # n_frames
        input_fdim = np.shape(x_train)[-1]  # n_mel

        # Dataloader

        train_loader = self.__setup_dataloader(x_train, y_train)

        # Instantiate model

        input_tdim = np.shape(x_train)[1]  # n_frames
        input_fdim = np.shape(x_train)[-1]  # n_mel

        # Choice of model here
        # ast_model = AST(input_tdim=input_tdim, n_classes=config.n_classes)
        # print(ast_model)

        if self.configs.model_name == 'conv':
            ast_model = Net
            print(ast_model)
        else:
            ast_model = AST(input_tdim=input_tdim, n_classes=self.configs.n_classes)
            print(ast_model)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Training on {device}')

        if torch.cuda.device_count() > 1:
            print("Using data parallel")
            ast_model = nn.DataParallel(ast_model, device_ids=list(range(torch.cuda.device_count())))

        ast_model = ast_model.to(device)

        ### Training loop

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(ast_model.parameters(), lr=self.configs.lr)

        all_train_loss = []
        all_train_acc = []
        all_val_loss = []
        all_val_acc = []
        best_val_loss = np.inf
        best_val_acc = -np.inf

        # best_train_loss = np.inf
        best_train_acc = -np.inf

        best_epoch = -1
        checkpoint_name = None
        overrun_counter = 0

        for e in range(self.configs.n_epochs):
            train_loss = 0.0
            ast_model.train()
            print(f'Training on {device}')

            all_y = []
            all_y_pred = []
            for batch_i, data in enumerate(train_loader, 0):

                ##Necessary in order to handle single and multi input feature spaces
                x, y = data

                x = x.to(device).detach()
                y = y.to(device).detach()

                if self.configs.model_name == 'conv':
                    x = torch.unsqueeze(x, dim=1)
                optimizer.zero_grad()
                y_pred = ast_model(x)
                loss = criterion(y_pred, torch.max(y, 1)[1])

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                all_y.append(y.cpu().detach())
                all_y_pred.append(y_pred.cpu().detach())

                del x
                del y

            all_train_loss.append(train_loss / len(train_loader))  # Note that with VAL train loader length changed

            all_y = torch.cat(all_y)
            all_y_pred = torch.cat(all_y_pred)
            train_acc = accuracy_score(np.argmax(all_y.detach().numpy(), axis=1),
                                       np.argmax(all_y_pred.detach().numpy(), axis=1))
            all_train_acc.append(train_acc)

            # Can add more conditions to support loss instead of accuracy. Use *-1 for loss inequality instead of acc
            if x_val:  # override val condition check
                val_acc = evaluate_model(ast_model, X_test, y_test)
                all_val_loss.append(val_loss)
                all_val_acc.append(val_acc)

                acc_metric = val_acc
                best_acc_metric = best_val_acc
            else:
                acc_metric = train_loss
                best_acc_metric = best_train_acc
            if acc_metric > best_acc_metric:
                # if checkpoint_name is not None:
                # os.path.join(os.path.pardir, 'models', 'pytorch', checkpoint_name)

                checkpoint_name = f'model_e{e}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pth'

                directory = os.path.join(self.configs.model_dir, str(cv_fold))
                if not os.path.isdir(directory):
                    os.mkdir(directory)
                print('Created directory:', directory)

                torch.save(ast_model.state_dict(), os.path.join(directory, checkpoint_name))
                print('Saving model to:', os.path.join(directory, checkpoint_name))
                best_epoch = e
                best_train_acc = train_acc
                best_train_loss = train_loss
                if x_val:  # override val loop
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                overrun_counter = -1

            overrun_counter += 1
            if x_val:  # override old convention for val detection
                print('Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, Val Acc: %.8f, overrun_counter %i' % (
                    e, train_loss / len(train_loader), train_acc, val_acc, overrun_counter))
            else:
                print('Epoch: %d, Train Loss: %.8f, Train Acc: %.8f, overrun_counter %i' % (
                    e, train_loss / len(train_loader), train_acc, overrun_counter))
            if overrun_counter > self.configs.max_overrun:
                break

        return ast_model, all_train_acc


    def evaluate(self):
        pass

    def test(self):
        pass


def evaluate_model(ast_model, cv_fold, filename):
    ''' Load data and evaluate model '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pickle_name_test = (
            'lms_cv_fold_' + str(cv_fold) + '_len_fft_' + str(config.len_fft) + '_win_len_' + str(config.win_len)
            + '_hop_len_' + str(config.hop_len) + '_n_mel_' + str(config.n_mel) + '.pkl')

    with open(os.path.join(config.dir_out_feat, pickle_name_test), 'rb') as f:
        feat_test = pickle.load(f)
        print('Loaded test features from:', os.path.join(config.dir_out_feat, pickle_name_test))
        X_test = feat_test['X_train']
        y_test = feat_test['y_train']
    ast_model.eval()

    test_loader = build_dataloader(X_test, y_test, shuffle=False)
    all_y_pred = []
    all_y = []
    with torch.no_grad():
        for x, y in test_loader:
            if config.debug:
                print('Ensuring correct dim', np.shape(x), np.shape(y))

            if config.model_name == 'conv':
                x = torch.unsqueeze(x, dim=1)
            x = x.to(device).detach()
            y = y.to(device).detach()

            y_pred = ast_model(x)
            # print(y_pred)
            all_y_pred.append(y_pred)
            all_y.append(y)

            del x
            del y
            del y_pred

        all_y_pred = torch.cat(all_y_pred).cpu().detach().numpy()
        all_y = torch.cat(all_y).cpu().detach().numpy()

        test_acc = accuracy_score(np.argmax(all_y, axis=1), np.argmax(all_y_pred, axis=1))
        print('Test accuracy', test_acc)
        print('Random guess', 1 / 50.)
        report = classification_report(np.argmax(all_y, axis=1), np.argmax(all_y_pred, axis=1), digits=5,
                                       output_dict=True)

        with open(os.path.join(config.plot_dir, filename + '_cm.txt'), "w") as text_file:
            print(report, file=text_file)

        return test_acc, report



if __name__ == "__main__":
    # Train and evaluate model according to ESC-50 five-fold validation scheme

    precision = []
    recall = []
    f1 = []

    for cv_fold in [1, 2, 3, 4, 5]:
        ast_model, all_train_acc = train_model(cv_fold)
        filename = ('e' + str(config.n_epochs) + 'num_heads' + str(config.num_heads)
                    + 'depth' + str(config.depth) + 'embed_dim' + str(config.embed_dim))
    #     # Evaluate best model according to early stopping criteria
    #     test_acc, report = evaluate_model(ast_model, cv_fold, filename)
    #     precision.append(report["macro avg"]["precision"])
    #     recall.append(report["macro avg"]["recall"])
    #     f1.append(report["macro avg"]["f1-score"])
    # print('Precision', np.mean(precision), '+-', np.std(precision))
    # print('Recall', np.mean(recall), '+-', np.std(recall))
    # print('F1', np.mean(f1), '+-', np.std(f1))
