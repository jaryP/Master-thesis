import torch.nn as nn
from abc import ABC, abstractmethod


class AbstractNetwork(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self._task = 0

    @abstractmethod
    def build_net(self, *args, **kwargs):
        pass

    @abstractmethod
    def eval_forward(self, x, task=None):
        pass

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        self._task = value

    @task.getter
    def task(self):
        return self._task

