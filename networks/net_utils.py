import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.init import normal_
from abc import ABC, abstractmethod
import torch.nn as nn


# ELU function (for initialization)
def elu(s):
    return np.maximum(0, s) + np.minimum(0, np.exp(s) - 1.0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class AbstractNetwork(ABC, nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.output_size = outputs
        self._task = 0

    @abstractmethod
    def build_net(self):
        pass

    @abstractmethod
    def eval_forward(self, x):
        pass

    @property
    def task(self):
        return self._task

    @task.setter
    def task(self, value):
        if value > self.output_size:
            value = self.output_size
        self._task = value

    @task.getter
    def task(self):
        return self._task


class KAF(nn.Module):
    """
    KAF.
    """

    def __init__(self, num_parameters, D=20, boundary=3.0, init_fcn=None, is_conv=False, trainable_dict=False):
        """
        :param num_parameters: number of neurons in the layer.
        :param D: size of the dictionary.
        :param boundary: range of the activation function.
        :param init_fcn: leave to None to initialize randomly, otherwise set a specific function for initialization.
        """

        super(KAF, self).__init__()
        self.num_parameters, self.D = num_parameters, D
        self.dict_numpy = np.linspace(-boundary, boundary, self.D).astype(np.float32).reshape(-1, 1)
        dict_tensor = torch.from_numpy(self.dict_numpy).view(-1)

        # Initialize the fixed dictionary
        if trainable_dict:
            dict_tensor = dict_tensor.unsqueeze(0)
            if is_conv:
                dict_tensor = dict_tensor.repeat(1, self.num_parameters, 1, 1, 1)
            else:
                dict_tensor = dict_tensor.repeat(1, self.num_parameters, 1)
            self.dict = Parameter(dict_tensor)
        else:
            self.register_buffer('dict', dict_tensor)

        # self.dict = Parameter(torch.from_numpy(self.dict_numpy)).view(-1)

        # Rule of thumb for gamma
        interval = (self.dict_numpy[1] - self.dict_numpy[0])
        sigma = 2 * interval  # empirically chosen
        self.gamma_init = float(0.5 / np.square(sigma))

        if is_conv:
            self.unsqueeze_dim = 4
            self.register_buffer('gamma',
                                 torch.from_numpy(np.ones((1, 1, 1, 1, self.D), dtype=np.float32) * self.gamma_init))
            self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, 1, 1, self.D))

        else:
            self.unsqueeze_dim = 2
            self.register_buffer('gamma', torch.from_numpy(np.ones((1, 1, self.D), dtype=np.float32) * self.gamma_init))
            self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, self.D))
            # self.alpha = Parameter(torch.FloatTensor(1, self.D))
            # print(self.alpha.size())

        # Initialization
        self.init_fcn = init_fcn
        if init_fcn is not None:
            K = np.exp(- self.gamma_init * (self.dict_numpy - self.dict_numpy.T) ** 2)
            self.alpha_init = np.linalg.solve(K + 1e-4 * np.eye(self.D), self.init_fcn(self.dict_numpy)).reshape(
                -1).astype(np.float32)
            if is_conv:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1, 1, 1)
            else:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1)
        else:
            self.alpha_init = None
            normal_(self.alpha.data, std=0.8)

        # Initialize the parameters

    def reset_parameters(self, is_conv=False):
        if self.init_fcn is not None:
            if is_conv:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1, 1, 1)
            else:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1)
        else:
            normal_(self.alpha.data, std=0.8)

    def forward(self, input):
        # First computes the Gaussian kernel
        K = torch.exp(- torch.mul((torch.add(input.unsqueeze(self.unsqueeze_dim), - self.dict)) ** 2, self.gamma))
        # print(K.size(), input.unsqueeze(self.unsqueeze_dim).size())
        y = torch.sum(K * self.alpha, self.unsqueeze_dim)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_parameters) + ')'
