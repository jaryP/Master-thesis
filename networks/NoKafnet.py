from torch import nn
import torch.nn.functional as F
from networks.net_utils import AbstractNetwork


class CNN(AbstractNetwork):
    def build_net(self):
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.output = nn.Linear(2048, self.output_size * 2)

    def __init__(self, num_tasks):
        super(CNN, self).__init__(outputs=num_tasks)
        self.build_net()

    def forward(self, input):
        x = F.relu(self.batchnorm1(self.conv1(input)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.maxpool1(F.relu(self.batchnorm3(self.conv3(x))))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        # x = self.maxpool(self.relu(self.batchnorm5(self.conv5(x))))
        x = self.output(x.reshape(input.shape[0], -1))
        return x[:, self.task * 2: self.task * 2 + 2]

    def eval_forward(self, x):
        return (nn.functional.softmax(self.forward(x), dim=1).max(dim=1)[1]).cpu().detach().numpy()


class MLP(AbstractNetwork):
    def __init__(self, num_task, hidden_size=400):
        super(MLP, self).__init__(num_task)
        self.build_net(hidden_size=hidden_size)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def build_net(self, *args, **kwargs):
        hidden_size = kwargs['hidden_size']
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.output_size)

    def eval_forward(self, x, task=None):
        x = self.forward(x)
        return (nn.functional.softmax(x, dim=1).max(dim=1)[1]).cpu().detach().numpy()
