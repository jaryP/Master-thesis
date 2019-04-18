import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.init import normal_
from abc import ABC, abstractmethod
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch import sigmoid as tsig
from copy import deepcopy


def elu(s):
    return np.maximum(0, s) + np.minimum(0, np.exp(s) - 1.0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def gaussian_kernel(model):
    def kernel(model, input):
        return torch.exp(-torch.mul((torch.add(input.unsqueeze(model.unsqueeze_dim), - model.dict)) ** 2, model.gamma))

    def kernel_init(x):
        return np.exp(- model.gamma_init * (x) ** 2)

    return kernel, kernel_init


def relu_kernel(model):
    def kernel(model, input):
        return F.relu(input.unsqueeze(model.unsqueeze_dim) - model.dict)

    def kernel_init(_):
        return None

    return kernel, kernel_init


def softplus_kernel(model):
    def kernel(model, input):
        return F.softplus(input.unsqueeze(model.unsqueeze_dim) - model.dict)

    def kernel_init(x):
        return np.log(np.exp(x) + 1.0)

    return kernel, kernel_init()


def polynomial_kernel(model):
    def kernel(model, input):
        return torch.pow(1 + torch.mul(input.unsqueeze(model.unsqueeze_dim), model.dict), 2)

    def kernel_init(x):
        return 1 + np.power(x, 2)

    return kernel, kernel_init


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

    def __init__(self, num_parameters, D=20, boundary=3.0, init_fcn=elu, is_conv=False,
                 trainable_dict=False, kernel='gaussian'):

        super(KAF, self).__init__()
        self.num_parameters, self.D = num_parameters, D
        self.dict_numpy = np.linspace(-boundary, boundary, self.D).astype(np.float32).reshape(-1, 1)

        dict_tensor = torch.from_numpy(self.dict_numpy).view(-1)

        if trainable_dict:
            dict_tensor = dict_tensor.unsqueeze(0)
            if is_conv:
                dict_tensor = dict_tensor.repeat(1, self.num_parameters, 1, 1, 1)
            else:
                dict_tensor = dict_tensor.repeat(1, self.num_parameters, 1)
            self.dict = Parameter(dict_tensor)
        else:
            self.register_buffer('dict', dict_tensor)

        interval = (self.dict_numpy[1] - self.dict_numpy[0])
        sigma = 2 * interval
        self.gamma_init = float(0.5 / np.square(sigma))

        if is_conv:
            self.unsqueeze_dim = 4
            self.register_buffer('gamma',
                                 torch.from_numpy(np.ones((1, 1, 1, 1, self.D), dtype=np.float32) * self.gamma_init))
        else:
            self.unsqueeze_dim = 2
            self.register_buffer('gamma', torch.from_numpy(np.ones((1, 1, self.D), dtype=np.float32) * self.gamma_init))

        if kernel == 'gaussian':
            self.kernel, self.kernel_init = gaussian_kernel(self)
        elif kernel == 'relu':
            self.kernel, self.kernel_init = relu_kernel(self)
        elif kernel == 'softplus':
            self.kernel, self.kernel_init = softplus_kernel(self)
        elif kernel == 'polynomial':
            self.kernel, self.kernel_init = polynomial_kernel(self)
        else:
            raise ValueError("Unexpected 'kernel'!", kernel)

        self.init_fcn = init_fcn
        if init_fcn is not None:

            K = self.kernel_init(self.dict_numpy - self.dict_numpy.T)

            if K is None:
                warnings.warn('Cannot perform kernel ridge regression with {} kernel '.format(kernel), RuntimeWarning)
                self.alpha_init = None
                normal_(self.alpha.data, std=0.8)
            else:
                alpha_init = np.linalg.solve(K + 1e-4 * np.eye(self.D), self.init_fcn(self.dict_numpy)).reshape(
                    -1).astype(np.float32)

                alpha_tensor = torch.from_numpy(alpha_init).view(-1)
                alpha_tensor = alpha_tensor.unsqueeze(0)

                if is_conv:
                    alpha_tensor = alpha_tensor.repeat(1, self.num_parameters, 1, 1, 1)
                else:
                    alpha_tensor = alpha_tensor.repeat(1, self.num_parameters, 1)

                self.alpha = Parameter(alpha_tensor)
        else:
            if is_conv:
                self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, 1, 1, self.D))
            else:
                self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, self.D))
            normal_(self.alpha.data, std=0.8)

    def forward(self, input):
        K = self.kernel(self, input)
        y = torch.sum(K * self.alpha, self.unsqueeze_dim)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_parameters) + ')'


class MultiKAF(nn.Module):
    def __init__(self, num_parameters, D=20, boundary=3.0, init_fcn=None, is_conv=False, trainable_dict=False,
                 kernel_combination='weighted'):
        super().__init__()
        self.kernels = ['gaussian', 'polynomial', 'relu']#, 'softplus']
        self.is_conv = is_conv

        self.dict_numpy = np.linspace(-boundary, boundary, D).astype(np.float32).reshape(-1, 1)
        dict_tensor = torch.from_numpy(self.dict_numpy).view(-1)

        if trainable_dict:
            dict_tensor = dict_tensor.unsqueeze(0)
            if is_conv:
                dict_tensor = dict_tensor.repeat(1, self.num_parameters, 1, 1, 1)
            else:
                dict_tensor = dict_tensor.repeat(1, self.num_parameters, 1)
            self.dict = Parameter(dict_tensor)
        else:
            self.register_buffer('dict', dict_tensor)

        K = 0
        inits = []
        for k in self.kernels:
            if k == 'gaussian':
                interval = (self.dict_numpy[1] - self.dict_numpy[0])
                sigma = 2 * interval
                self.gamma_init = float(0.5 / np.square(sigma))

                kernel, kernel_init = gaussian_kernel(self)
                inits.append(kernel_init)

                if is_conv:
                    self.register_buffer('gamma',
                                         torch.from_numpy(np.ones((1, 1, 1, 1, D), dtype=np.float32) * self.gamma_init))
                    # self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, 1, 1, self.D))
                else:
                    self.register_buffer('gamma',
                                         torch.from_numpy(np.ones((1, 1, D), dtype=np.float32) * self.gamma_init))
            elif k == 'relu':
                kernel, _ = relu_kernel(self)

            elif k == 'softplus':
                kernel, kernel_init = softplus_kernel(self)
                inits.append(kernel_init)

            elif k == 'polynomial':
                kernel, kernel_init = polynomial_kernel(self)
                inits.append(kernel_init)

            else:
                raise ValueError("Unexpected 'kernel'!", k)

            setattr(self, k, kernel)

        alpha = torch.Tensor(D).view(-1)
        # alpha.fill_(1/D)
        # normal_(alpha.data, std=0.2)
        # self.gamma_init = None
        if init_fcn is not None:

            if isinstance(K, int):
                warnings.warn('Cannot perform kernel ridge regression with {} kernel '
                              .format(self.kernels), RuntimeWarning)
                alpha_init = None
                # normal_(self.alpha.data, std=0.8)
            else:
                # K /= len(self.kernels)
                alpha_init = np.linalg.solve(K + 1e-4 * np.eye(D), init_fcn(self.dict_numpy)).reshape(
                    -1).astype(np.float32)
                alpha.data = torch.from_numpy(alpha_init)
                # if is_conv:
                #     self.alpha.data = torch.from_numpy(alpha_init).repeat(1, self.num_parameters, 1, 1, 1)
                # else:
                #     self.alpha.data = torch.from_numpy(alpha_init).repeat(1, self.num_parameters, 1)

        mu = torch.Tensor(len(self.kernels)).view(-1)
        mu.fill_(1/len(self.kernels))

        if is_conv:
            self.unsqueeze_dim = 4
            alpha = alpha.repeat(1, num_parameters, 1, 1, 1)
            mu = mu.repeat(1, num_parameters, 1, 1, 1)
        else:
            self.unsqueeze_dim = 2
            alpha = alpha.repeat(1, num_parameters, 1)
            mu = mu.repeat(1, num_parameters, 1)

        self.alpha = alpha
        self.mu = Parameter(mu, requires_grad=True)

        if init_fcn is None:
            normal_(alpha, mean=0, std=0.2)

        if kernel_combination == 'softmax':
            self.kernel_combination = self.kernel_softmax

        elif kernel_combination == 'sigmoid':
            self.kernel_combination = self.kernel_sigmoid

        elif kernel_combination == 'layer_attention':
            self.attention = nn.Linear(num_parameters, 1, bias=False)
            self.kernel_combination = self.kernel_layer_attention

        elif kernel_combination == 'neuron_attention':
            self.attention = nn.Linear(num_parameters, num_parameters, bias=False)
            self.kernel_combination = self.kernel_neuron_attention

        elif kernel_combination == 'sum' or kernel_combination == 'weighted':
            self.kernel_combination = self.kernel_sum

        else:
            warnings.warn('Kernel combination strategy not understood {}. '
                          'Default weighted sum will be used '.format(kernel_combination), RuntimeWarning)
            self.kernel_combination = self.kernel_sum

        self.alpha = Parameter(alpha, requires_grad=True)

    def kernel_sum(self, x, f):
        f = self.mu.unsqueeze(-2) * f
        y = torch.sum(f, self.unsqueeze_dim+1)
        return y

    def kernel_softmax(self, x, f):
        sm = F.softmax(self.mu, dim=-1)

        f = torch.mul(sm.unsqueeze(-2), f)
        y = torch.sum(f, self.unsqueeze_dim+1)
        return y

    def kernel_sigmoid(self, x, f):
        sm = tsig(self.mu)
        f = torch.mul(sm.unsqueeze(-2), f)
        y = torch.sum(f, self.unsqueeze_dim+1)
        return y

    def kernel_layer_attention(self, x, f):
        s = self.attention(x)
        d = [1] * len(tuple(self.mu.size()))
        d[0] = s.size()[0]
        mu = self.mu.repeat(d)
        for _ in range(2, len(d)):
            s.unsqueeze_(-1)
        d = list(mu.size())
        d[0] = -1
        s = s.expand(d)
        mu = torch.mul(mu, s)
        sm = F.softmax(mu.unsqueeze(-2), dim=-1)
        f = torch.mul(sm, f)
        y = torch.sum(f, self.unsqueeze_dim+1)
        return y

    def kernel_neuron_attention(self, x, f):
        s = self.attention(x)
        d = [1] * len(tuple(self.mu.size()))
        d[0] = s.size()[0]
        mu = self.mu.repeat(d)
        for _ in range(2, len(d)):
            s.unsqueeze_(-1)
        d = list(mu.size())
        d[0] = -1
        s = s.expand(d)
        mu = torch.mul(mu, s)
        sm = F.softmax(mu.unsqueeze(-2), dim=-1)
        f = torch.mul(sm, f)
        y = torch.sum(f, self.unsqueeze_dim+1)
        return y

    def forward(self, x):
        f = []
        for k in self.kernels:
            c = getattr(self, k)(self, x)
            f.append(c)

        f = torch.stack(f, -1)
        f = f.to(self.mu.device)
        y = self.kernel_combination(x, f)
        y = torch.mul(y, self.alpha)
        y = torch.sum(y, self.unsqueeze_dim)

        return y
