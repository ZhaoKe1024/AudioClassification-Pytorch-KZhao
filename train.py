#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/15 17:41
# @Author: ZhaoKe
# @File : train.py
# @Software: PyCharm
from ackit.trainer_ConvEncoder import TrainerEncoder


def main():
    trainer = TrainerEncoder(configs="./configs/autoencoder.yaml", istrain=False)
    trainer.train_encoder()


if __name__ == '__main__':
    main()
