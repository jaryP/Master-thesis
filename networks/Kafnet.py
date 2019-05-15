from networks.net_utils import KAF, elu, MultiKAF
import torch.nn as nn
from networks.net_utils import AbstractNetwork


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
        pass


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

