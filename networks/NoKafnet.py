from torch import nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_tasks):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(128)
        # self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.batchnorm5 = nn.BatchNorm2d(128)
        self.output = nn.Linear(2048, num_tasks * 2)
        self.current_task = 0

    def forward(self, input):
        x = F.relu(self.batchnorm1(self.conv1(input)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.maxpool1(F.relu(self.batchnorm3(self.conv3(x))))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        # x = self.maxpool(self.relu(self.batchnorm5(self.conv5(x))))
        x = self.output(x.reshape(input.shape[0], -1))
        return x[:, self.current_task * 2: self.current_task * 2 + 2]

    def set_task(self, task):
        self.current_task = torch.tensor(task)

    def eval_forward(self, x):
        return (nn.functional.softmax(x, dim=1).max(dim=1)[1]).cpu().detach().numpy()