#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/19 10:32
# @Author: ZhaoKe
# @File : classifiers.py
# @Software: PyCharm
import torch.nn as nn
from ackit.modules.attentions import *


class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        self.layers = nn.Linear(in_dim, num_classes)

    def forward(self, features):
        scores = self.layers(features)
        return scores


class LSTM_Classifier(nn.Module):
    def __init__(self, inp_size, hidden_size, n_classes):
        super(LSTM_Classifier, self).__init__()
        self.lstm = nn.LSTM(inp_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.squeeze()
        lstm_out, (hidden, _) = self.lstm(x.transpose(1, 2))
        lstm_out = lstm_out[:, -1, :]
        out = self.classifier(lstm_out)
        return out


class LSTM_Attn_Classifier(nn.Module):
    def __init__(self, inp_size, hidden_size, n_classes, return_attn_weights=False, attn_type='dot'):
        super(LSTM_Attn_Classifier, self).__init__()
        self.return_attn_weights = return_attn_weights
        self.lstm = nn.LSTM(inp_size, hidden_size, batch_first=True)
        self.attn_type = attn_type

        if self.attn_type == 'dot':
            self.attention = DotAttention()
        elif self.attn_type == 'soft':
            self.attention = SoftAttention(hidden_size, hidden_size)

        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.squeeze()
        lstm_out, (hidden, _) = self.lstm(x.transpose(1, 2))

        if self.attn_type == 'dot':
            attn_output = self.attention(lstm_out, hidden)
            attn_weights = self.attention.get_weights(lstm_out, hidden)
        elif self.attn_type == 'soft':
            attn_output = self.attention(lstm_out)
            attn_weights = self.attention.get_weights(lstm_out)

        out = self.classifier(attn_output)
        if self.return_attn_weights:
            return out, attn_weights
        else:
            return out


def test_lstm():
    # input_size: 时间步
    # hidden_size:
    # num_layer: 层数
    x = torch.randn(16, 64, 128)  # (bs, length, dim)
    lstm1 = LSTM_Classifier(inp_size=64, hidden_size=128, n_classes=2)
    lstm2 = LSTM_Attn_Classifier(inp_size=64, hidden_size=128, n_classes=2,
                                 return_attn_weights=False, attn_type="soft")
    print(lstm1(x).shape)
    print(lstm2(x).shape)


if __name__ == '__main__':
    # main()
    test_lstm()
