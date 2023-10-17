#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/15 17:41
# @Author: ZhaoKe
# @File : train_hst.py
# @Software: PyCharm
import argparse

from ackit.trainer_hst import HSTTrainer
from ackit.utils import utils


def main():
    trainer = HSTTrainer(configs="./configs/tdnn.yaml", use_gpu=True)
    trainer.train()
    # trainer.print_configs()
    # if not args.load_epoch:
    #     trainer.train()


# def run():
#     params = utils.load_yaml("configs/hst.yaml")
#     parser = argparse.ArgumentParser(description=params['description'])
#     for key, value in params.items():
#         parser.add_argument(f'--{key}', default=value, type=utils.set_type)
#     args = parser.parse_args()
#     main(args=args)


if __name__ == '__main__':
    main()
