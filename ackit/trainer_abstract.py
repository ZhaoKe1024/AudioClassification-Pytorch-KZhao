#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/22 16:46
# @Author: ZhaoKe
# @File : trainer_abstract.py
# @Software: PyCharm
from abc import ABC, abstractmethod

import yaml

from ackit.utils.utils import dict_to_object


class Trainer(ABC):
    def __init__(self, config):
        self.configs = None
        if isinstance(config, str):
            with open(config, 'r') as jsf:
                cfg = yaml.load(jsf.read(), Loader=yaml.FullLoader)
                self.configs = dict_to_object(cfg)

    @abstractmethod
    def _setup_dataloader(self):
        pass

    @abstractmethod
    def _setup_model(self):
        pass

    @abstractmethod
    def _train_epoch(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def test(self):
        pass
