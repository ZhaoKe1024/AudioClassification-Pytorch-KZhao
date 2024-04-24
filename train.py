#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/15 17:41
# @Author: ZhaoKe
# @File : train.py
# @Software: PyCharm
from ackit.trainer_ConvEncoder import TrainerEncoder
from ackit.coughcls_tdnn import TrainerTDNN


def run_encoder_classification():
    trainer = TrainerEncoder(configs="./configs/autoencoder.yaml", istrain=False)
    trainer.train_encoder()


def run_tdnn_classification():
    # istrain设为False，则不创建输出的文件夹，并且只读取一点数据用来测试，测试是否有bug。
    trainer = TrainerTDNN(configs="./configs/tdnn_coughvid.yaml", istrain=True, isdemo=False)
    trainer.train()


def test():
    load_epoch = 14
    resume_model = f"./runs/tdnn/202401162007_tdnn/model_epoch_{load_epoch}"
    trainer = TrainerTDNN(configs="./configs/autoencoder.yaml", istrain=False)
    trainer.test_tsne(resume_model_path=resume_model, load_epoch=load_epoch)


if __name__ == '__main__':
    run_tdnn_classification()
    # test()
