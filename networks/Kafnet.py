from networks.net_utils import KAF, elu, MultiKAF, leakyRelu
from torch import nn, from_numpy
from networks.net_utils import AbstractNetwork, Flatten, CustomLinear
import torch.nn.functional as F
import numpy as np


class VGG(AbstractNetwork):

    def __init__(self, out_dim, D=10, kernel='softplus', trainable_dict=False, alpha_mean=0, alpha_std=0.8,
                 boundary=3, positive_dict=False, init_fcn=elu):

        super(VGG, self).__init__(outputs=out_dim, )

        self.layers = [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M']
        #         self.layers = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        self.features = self.build_net(D=D, kernel=kernel, trainable_dict=trainable_dict, alpha_mean=alpha_mean,
                                       alpha_std=alpha_std, boundary=boundary, positive_dict=positive_dict,
                                       init_fcn=init_fcn)
        # self.classifier = nn.Linear(2048, 10)
        self.features_processing = self.classifier = nn.Sequential(
            CustomLinear(4096 // 2),
            nn.ReLU(inplace=True),
            # KAF(4096 // 2, D=D, kernel=kernel, is_conv=False, trainable_dict=trainable_dict,
            #     boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn),
            nn.Dropout(),
            nn.Linear(4096 // 2, 4096 // 2),
            nn.ReLU(inplace=True),
            # KAF(4096 // 2, D=D, kernel=kernel, is_conv=False, trainable_dict=trainable_dict,
            #     boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn),
            nn.Dropout())

        self.classifier = CustomLinear(self.output_size)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(self.features_processing(out))
        if isinstance(self._task, (list, tuple, set)):
            mask = np.zeros(self.output_size)
            for i in self._task:
                mask[i] = 1
            out = out * from_numpy(mask).float().to(out.device)
        return out

    def eval_forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(self.features_processing(out))
        return nn.functional.softmax(out, dim=1).max(dim=1)[1]

    def build_net(self, D=10, kernel='softplus', trainable_dict=False, alpha_mean=0, alpha_std=0.8,
                  boundary=3, positive_dict=False, init_fcn=None):
        layers = []
        in_channels = 3
        for x in self.layers:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    # nn.ReLU(inplace=True)
                    KAF(x, D=D, kernel=kernel, is_conv=True, trainable_dict=trainable_dict,
                        boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn)
                ]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)

    def embedding(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.features_processing(out)
        return out


class CNN(AbstractNetwork):

    def build_net(self, D=10, kernel='softplus', trainable_dict=False, alpha_mean=0, alpha_std=0.8,
                  boundary=3, positive_dict=False, init_fcn=None):

        layers = []
        in_channels = 3
        for x in self.topology:
            if x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    nn.Dropout(0.25)
                ]
            elif x == 'BN':
                layers += [
                    nn.BatchNorm2d(in_channels)
                ]
            else:
                x = x
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=1),
                    # nn.ReLU(inplace=True),
                    KAF(x, D=D, kernel=kernel, is_conv=True, trainable_dict=trainable_dict, alpha_mean=alpha_mean,
                        boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn, alpha_std=alpha_std)
                ]
                in_channels = x

        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        layers = nn.Sequential(*layers)

        return layers

    def __init__(self, num_tasks, D=10, kernel='softplus', trainable_dict=False, alpha_mean=0, alpha_std=0,
                 boundary=3, positive_dict=False, init_fcn=elu, topology=None, incremental=False):
        super(CNN, self).__init__(outputs=num_tasks)

        if topology is None:
            topology = [32, 32, 'M', 64, 64, 'M']

        self.topology = topology
        self.incremental = incremental

        self.features = self.build_net(D=D, kernel=kernel, trainable_dict=trainable_dict, alpha_std=alpha_std,
                                       alpha_mean=alpha_mean,
                                       boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn)

        # self.features_processing = nn.Sequential(CustomLinear(512),
        #                                          KAF(num_parameters=512, D=D, kernel=kernel, is_conv=False,
        #                                              trainable_dict=trainable_dict,
        #                                              boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn),
        #                                          nn.BatchNorm1d(512),
        #                                          nn.Dropout(0.5)
        #                                          )

        self.classification_layer = CustomLinear(self.output_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.features_processing(x)
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

        return x

    def eval_forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = self.features_processing(x)
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
        # x = self.features_processing(x)

        return x


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

        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

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

        # self.features_processing = nn.Sequential(CustomLinear(512),
        #                                          KAF(num_parameters=512, D=D, kernel=kernel, is_conv=False,
        #                                              trainable_dict=trainable_dict,
        #                                              boundary=boundary, positive_dict=positive_dict, init_fcn=init_fcn),
        #                                          nn.BatchNorm1d(512),
        #                                          nn.Dropout(0.5)
        #                                          )

        self.classification_layer = None
        # self.classification_layer = nn.Linear(1584, self.output_size)

    def forward(self, x):
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
        # x = self.features_processing(x)

        return x


class KAFCNN(AbstractNetwork):

    def __init__(self, num_tasks):
        super(KAFCNN, self).__init__(outputs=num_tasks)
        # self.kaf5 = KAF(96, init_fcn=elu, is_conv=True)
        self.build_net()

    def forward(self, input):
        x = self.kaf1(self.batchnorm1(self.conv1(input)))
        x = self.kaf2(self.batchnorm2(self.conv2(x)))
        x = self.maxpool(self.kaf3(self.batchnorm3(self.conv3(x))))
        x = self.kaf4(self.batchnorm4(self.conv4(x)))
        # x = self.maxpool(self.kaf5(self.batchnorm5(self.conv5(x))))
        x = self.output(x.reshape(input.shape[0], -1))
        return x[:, self.task * 2: self.task * 2 + 2]

    def build_net(self):
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 52, kernel_size=3, padding=1, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(52)
        self.conv3 = nn.Conv2d(52, 52, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(52)
        self.conv4 = nn.Conv2d(52, 96, kernel_size=3, padding=1, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(96)
        self.kaf1 = KAF(32, init_fcn=elu, is_conv=True, D=15)
        self.kaf2 = KAF(52, init_fcn=elu, is_conv=True, D=15)
        self.kaf3 = KAF(52, init_fcn=elu, is_conv=True, D=15)
        self.kaf4 = KAF(96, init_fcn=elu, is_conv=True, D=15)
        self.output = nn.Linear(1536, self.output_size * 2)

    # def set_task(self, task):
    #     self.current_task = task

    def eval_forward(self, x):
        return (nn.functional.softmax(self.forward(x), dim=1).max(dim=1)[1]).cpu().detach().numpy()

    def embedding(self, x):
        x = self.kaf1(self.batchnorm1(self.conv1(x)))
        x = self.kaf2(self.batchnorm2(self.conv2(x)))
        x = self.maxpool(self.kaf3(self.batchnorm3(self.conv3(x))))
        x = self.kaf4(self.batchnorm4(self.conv4(x)))
        x = x.reshape(input.shape[0], -1)
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
