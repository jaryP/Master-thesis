from networks.net_utils import KAF, elu, MultiKAF, leakyRelu
from torch import nn, from_numpy
from networks.net_utils import AbstractNetwork, Flatten, CustomLinear
import torch.nn.functional as F
import numpy as np


class synCNN(AbstractNetwork):

    def build_net(self, D=10, kernel='softplus', trainable_dict=False, alpha_mean=0, alpha_std=0.8,
                  boundary=3, positive_dict=False, init_fcn=None):

        layers = []
        in_channels = 3
        for x in self.topology:
            layers += [
                nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=1),
                KAF(x, D=D, kernel=kernel, is_conv=True, trainable_dict=trainable_dict, alpha_mean=alpha_mean,
                    boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn, alpha_std=alpha_std),
                nn.Conv2d(x, x, kernel_size=3, padding=0, stride=1),
                KAF(x, D=D, kernel=kernel, is_conv=True, trainable_dict=trainable_dict, alpha_mean=alpha_mean,
                    boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn, alpha_std=alpha_std),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout(0.25)
            ]
            in_channels = x

        layers = nn.Sequential(*layers)

        return layers

    def __init__(self, num_tasks, D=10, kernel='softplus', trainable_dict=False, alpha_mean=0, alpha_std=0,
                 boundary=3, positive_dict=False, init_fcn=elu, topology=None, incremental=False):

        super().__init__(outputs=num_tasks)

        if topology is None:
            topology = [32,  64]

        self.topology = topology
        self.incremental = incremental

        self.features = self.build_net(D=D, kernel=kernel, trainable_dict=trainable_dict, alpha_std=alpha_std,
                                       alpha_mean=alpha_mean,
                                       boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn)

        self.classification_layer = None

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.classification_layer is None:
            self.classification_layer = nn.Linear(x.size()[1], self.output_size).to(x.device)

        x = self.classification_layer(x)

        mask = np.zeros(self.output_size)
        if self.incremental:
            for i in self.used_tasks:
                mask[i] = 1
        else:
            if isinstance(self._task, (list, tuple, set)):
                for i in self._task:
                    mask[i] = 1

        if mask.sum() != 0:
            x = x * from_numpy(mask).float().to(x.device)
        return x

    def eval_forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.features_processing(x)
        if self.classification_layer is None:
            self.classification_layer = nn.Linear(x.size()[1], self.output_size).to(x.device)

        x = self.classification_layer(x)

        mask = np.zeros(self.output_size)
        if self.incremental:
            for i in self._used_tasks:
                mask[i] = 1
        else:
            if isinstance(self._task, (list, tuple, set)):
                for i in self._task:
                    mask[i] = 1

        if mask.sum() != 0:
            x = x * from_numpy(mask).float().to(x.device)

        return nn.functional.softmax(x, dim=1).max(dim=1)[1]

    def embedding(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        return x


class KAFMLP(AbstractNetwork):

    def __init__(self, n_outputs, hidden_size=400, kaf_init_fcn=None, trainable_dict=False, kernel='gaussian', D=20):
        super(KAFMLP, self).__init__(n_outputs)
        self.build_net(hidden_size=hidden_size, kaf_init_fcn=kaf_init_fcn, trainable_dict=trainable_dict,
                       kernel=kernel, D=D)

    def build_net(self, *args, **kwargs):
        hidden_size = kwargs['hidden_size']
        kaf_init_fcn = kwargs['kaf_init_fcn']
        trainable_dict = kwargs.get('trainable_dict', False)
        kernel = kwargs.get('kernel')
        D = kwargs.get('D')

        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.output_size)
        self.kaf1 = KAF(hidden_size, init_fcn=kaf_init_fcn, D=D, trainable_dict=trainable_dict, kernel=kernel)
        self.kaf2 = KAF(hidden_size, init_fcn=kaf_init_fcn, D=D, trainable_dict=trainable_dict, kernel=kernel)
        self.kaf3 = KAF(hidden_size, init_fcn=kaf_init_fcn, D=D, trainable_dict=trainable_dict, kernel=kernel)

    def eval_forward(self, x, task=None):
        x = self.forward(x)
        return (nn.functional.softmax(x, dim=1).max(dim=1)[1]).cpu().detach().numpy()

    def forward(self, x):
        x = self.kaf1(self.fc1(x))
        x = self.kaf2(self.fc2(x))
        x = self.kaf3(self.fc3(x))
        x = self.fc4(x)
        return x

    def embedding(self, x):
        x = self.kaf1(self.fc1(x))
        x = self.kaf2(self.fc2(x))
        x = self.kaf3(self.fc3(x))
        return x


class MultiKAFMLP(AbstractNetwork):
    def __init__(self, n_outputs, hidden_size=400, kaf_init_fcn=None, trainable_dict=False, D=20,
                 kernel_combination='weighted', kernels=None):
        super().__init__(n_outputs)

        self.build_net(hidden_size=hidden_size, kaf_init_fcn=kaf_init_fcn, trainable_dict=trainable_dict,
                       D=D, kernel_combination=kernel_combination, kernels=kernels)

    def build_net(self, *args, **kwargs):
        hidden_size = kwargs['hidden_size']
        kaf_init_fcn = kwargs['kaf_init_fcn']
        trainable_dict = kwargs.get('trainable_dict', False)
        D = kwargs.get('D')
        kernel_combination = kwargs.get('kernel_combination')
        kernels = kwargs.get('kernels')

        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.output_size)

        self.kaf1 = MultiKAF(hidden_size, init_fcn=kaf_init_fcn, D=D, trainable_dict=trainable_dict,
                             kernel_combination=kernel_combination, kernels=kernels)
        self.kaf2 = MultiKAF(hidden_size, init_fcn=kaf_init_fcn, D=D, trainable_dict=trainable_dict,
                             kernel_combination=kernel_combination, kernels=kernels)
        self.kaf3 = MultiKAF(hidden_size, init_fcn=kaf_init_fcn, D=D, trainable_dict=trainable_dict,
                             kernel_combination=kernel_combination, kernels=kernels)

        # self.kaf1 = KAF(hidden_size, init_fcn=kaf_init_fcn, D=D, trainable_dict=trainable_dict, kernel='gaussian')
        # self.kaf2 = KAF(hidden_size, init_fcn=kaf_init_fcn, D=D, trainable_dict=trainable_dict, kernel='gaussian')
        # self.kaf3 = KAF(hidden_size, init_fcn=kaf_init_fcn, D=D, trainable_dict=trainable_dict, kernel='gaussian')

    def eval_forward(self, x, task=None):
        x = self.forward(x)
        return (nn.functional.softmax(x, dim=1).max(dim=1)[1]).cpu().detach().numpy()

    def forward(self, x):
        x = self.kaf1(self.fc1(x))
        x = self.kaf2(self.fc2(x))
        x = self.kaf3(self.fc3(x))
        x = self.fc4(x)
        return x

    def embedding(self, x):
        x = self.kaf1(self.fc1(x))
        x = self.kaf2(self.fc2(x))
        x = self.kaf3(self.fc3(x))
        return x
