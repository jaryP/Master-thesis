from torch import nn, from_numpy
import torch.nn.functional as F
from networks.net_utils import AbstractNetwork, Flatten, CustomLinear
import numpy as np


class VGG(AbstractNetwork):
    def __init__(self, out_dim):
        super(VGG, self).__init__(outputs=out_dim)
        self.layers = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.features = self.build_net()
        # self.classifier = nn.Linear(2048, 10)

        self.features_processing = self.classifier = nn.Sequential(
            CustomLinear(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            )

        self.classifier = CustomLinear(self.output_size)

    def build_net(self):
        layers = []
        in_channels = 3
        for x in self.layers:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        return nn.Sequential(*layers)

    def embedding(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.features_processing(out)
        return out

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


class CNN(AbstractNetwork):

    def build_net(self):

        layers = []
        in_channels = 3
        for x in self.topology:
            if x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
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
                    nn.ReLU(inplace=True),
                ]
                in_channels = x

        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        layers = nn.Sequential(*layers)

        return layers

    def __init__(self, num_tasks, topology=None):
        super(CNN, self).__init__(outputs=num_tasks)

        if topology is None:
            topology = [32, 32, 'M', 64, 64, 'M']

        self.topology = topology

        self.features = self.build_net()

        self.features_processing = nn.Sequential(CustomLinear(512),
                                                 nn.ReLU(True),
                                                 nn.Dropout(0.5)
                                                 )

        self.classification_layer = CustomLinear(self.output_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.features_processing(x)
        x = self.classification_layer(x)

        if isinstance(self._task, (list, tuple, set)):
            mask = np.zeros(self.output_size)
            for i in self._task:
                mask[i] = 1
            x = x * from_numpy(mask).float().to(x.device)

        return x

    def eval_forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.features_processing(x)
        x = self.classification_layer(x)

        return nn.functional.softmax(x, dim=1).max(dim=1)[1]

    def embedding(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.features_processing(x)

        return x


class synCNN(AbstractNetwork):

    def build_net(self):

        layers = []
        in_channels = 3
        for x in self.topology:
            layers += [
                nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(x, x, kernel_size=3, padding=0, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Dropout(0.25),
            ]
            in_channels = x

        layers = nn.Sequential(*layers)

        return layers

    def __init__(self, num_tasks, topology=None, incremental=False):
        super().__init__(outputs=num_tasks)

        if topology is None:
            topology = [32, 64]

        self.topology = topology
        self.incremental = incremental

        self.features = self.build_net()

        self.classification_layer = nn.Linear(2304, self.output_size)

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



class MLP(AbstractNetwork):
    def __init__(self, num_task, hidden_size=400):
        super(MLP, self).__init__(num_task)
        self.build_net(hidden_size=hidden_size)

    def build_net(self, *args, **kwargs):
        hidden_size = kwargs['hidden_size']
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.output_size)

    def eval_forward(self, x, task=None):
        x = self.forward(x)
        return (nn.functional.softmax(x, dim=1).max(dim=1)[1]).cpu().detach().numpy()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def embedding(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
