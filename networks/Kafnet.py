import torch.nn as nn

from networks.ActivationFunctions import KAF, elu


class KAFCNN(nn.Module):

    def __init__(self, num_tasks):
        super(KAFCNN, self).__init__()
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
        # self.kaf5 = KAF(96, init_fcn=elu, is_conv=True)
        self.output = nn.Linear(1536, num_tasks * 2)
        self.current_task = 0

    def forward(self, input):
        x = self.kaf1(self.batchnorm1(self.conv1(input)))
        x = self.kaf2(self.batchnorm2(self.conv2(x)))
        x = self.maxpool(self.kaf3(self.batchnorm3(self.conv3(x))))
        x = self.kaf4(self.batchnorm4(self.conv4(x)))
        # x = self.maxpool(self.kaf5(self.batchnorm5(self.conv5(x))))
        x = self.output(x.reshape(input.shape[0], -1))
        return x[:, self.current_task * 2:self.current_task * 2 + 2]

    def set_task(self, task):
        self.current_task = task

    def eval_forward(self, x):
        return (nn.functional.softmax(x, dim=1).max(dim=1)[1]).cpu().detach().numpy()
