import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.init import normal_
from abc import ABC, abstractmethod
import torch.nn as nn
import torch.nn.functional as F
import warnings


def elu(s):
    return np.maximum(0, s) + np.minimum(0, np.exp(s) - 1.0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def gaussian_kernel(model):
    def kernel(input):
        return torch.exp(- torch.mul((torch.add(input.unsqueeze(model.unsqueeze_dim), - model.dict)) ** 2, model.gamma))

    return kernel, np.exp(- model.gamma_init * (model.dict_numpy - model.dict_numpy.T) ** 2)


def relu_kernel(model):
    def kernel(input):
        return F.relu(input.unsqueeze(model.unsqueeze_dim) - model.dict)

    return kernel, None


def softplus_kernel(model):
    def kernel(input):
        return F.softplus(input.unsqueeze(model.unsqueeze_dim) - model.dict)

    return kernel, np.log(np.exp(model.dict_numpy - model.dict_numpy.T) + 1.0)


def polynomial_kernel(model):
    def kernel(input):
        return torch.pow(1 + torch.mul(input.unsqueeze(model.unsqueeze_dim), model.dict), 2)

    return kernel, (1 + np.power(model.dict_numpy - model.dict_numpy.T, 2))


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
            self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, 1, 1, self.D))
        else:
            self.unsqueeze_dim = 2
            self.register_buffer('gamma', torch.from_numpy(np.ones((1, 1, self.D), dtype=np.float32) * self.gamma_init))
            self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, self.D))

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
            K = self.kernel_init

            if K is None:
                warnings.warn('Cannot perform kernel ridge regression with {} kernel '.format(kernel), RuntimeWarning)
                self.alpha_init = None
                normal_(self.alpha.data, std=0.8)
            else:
                self.alpha_init = np.linalg.solve(K + 1e-4 * np.eye(self.D), self.init_fcn(self.dict_numpy)).reshape(
                    -1).astype(np.float32)
                if is_conv:
                    self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1, 1, 1)
                else:
                    self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1)
        else:
            self.alpha_init = None
            normal_(self.alpha.data, std=0.8)

    def reset_parameters(self, is_conv=False):
        if self.init_fcn is not None:
            if is_conv:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1, 1, 1)
            else:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1)
        else:
            normal_(self.alpha.data, std=0.8)

    def forward(self, input):
        K = self.kernel(input)
        y = torch.sum(K * self.alpha, self.unsqueeze_dim)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.num_parameters) + ')'


class MultiKAF(nn.Module):
    def __init__(self, num_parameters, D=20, boundary=3.0, init_fcn=None, is_conv=False, trainable_dict=False,
                 kernel_combination='weighted'):
        super().__init__()
        self.kernels = ['gaussian', 'polynomial']#, 'relu', 'softplus']

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
        for k in self.kernels:
            if k == 'gaussian':
                interval = (self.dict_numpy[1] - self.dict_numpy[0])
                sigma = 2 * interval
                self.gamma_init = float(0.5 / np.square(sigma))

                kernel, kernel_init = gaussian_kernel(self)
                K += kernel_init

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
                K += kernel_init

            elif k == 'polynomial':
                kernel, kernel_init = polynomial_kernel(self)
                K += kernel_init

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
            mu = mu.repeat(1, num_parameters, D, 1, 1, 1)
        else:
            self.unsqueeze_dim = 2
            alpha = alpha.repeat(1, num_parameters, 1)
            mu = mu.repeat(1, num_parameters, D, 1)

        self.alpha = alpha
        self.mu = Parameter(mu, requires_grad=True)

        if init_fcn is None:
            normal_(alpha, mean=0, std=0.2)

        if kernel_combination == 'weighted':
            self.kernel_combination = self.kernel_sum
        elif kernel_combination == 'softmax':
            self.kernel_combination = self.kernel_softmax
        elif kernel_combination == 'normalized':
            self.kernel_combination = self.kernel_normalized_sum
        elif kernel_combination == 'sum':
            self.kernel_combination = self.kernel_sum
            self.alpha.fill_(1/D)
        else:
            warnings.warn('Kernel aggregator not understood {}. '
                          'Default weighted sum will be used '.format(kernel_combination), RuntimeWarning)
            self.kernel_combination = self.kernel_sum

        self.alpha = Parameter(alpha, requires_grad=True)

    def kernel_sum(self, x):
        x = torch.sum(x, self.unsqueeze_dim+1)
        x = self.alpha * x
        y = torch.sum(x, self.unsqueeze_dim)
        return y

    def kernel_normalized_sum(self, x):
        x = torch.sum(x, self.unsqueeze_dim+1)
        f = F.softmax(self.alpha, dim=self.unsqueeze_dim) * x
        y = torch.sum(f, self.unsqueeze_dim)
        return y

    def kernel_softmax(self, x):
        x = torch.sum(x, self.unsqueeze_dim+1)
        sm = F.softmax(self.alpha, dim=self.unsqueeze_dim)
        f = torch.mul(sm, x)
        y = torch.sum(f, self.unsqueeze_dim)
        return y

    def forward(self, x):
        f = []
        for k in self.kernels:
            c = getattr(self, k)(x)
            f.append(c)
        f = torch.stack(f, -1)
        f = f.to(self.mu.device)
        f = torch.mul(f, self.mu)
        y = self.kernel_combination(f)
        return y
