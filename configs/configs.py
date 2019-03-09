import torch
from torch import nn, optim, utils


class DefaultConfig(object):
    LR = 0.001
    L1_REG = 1e-4

    ITERS = 1
    EPOCHS = 15
    BATCH_SIZE = 12
    IS_CONVOLUTIONAL = True

    EWC_SAMPLE_SIZE = 250
    EWC_IMPORTANCE = 1000
    USE_EWC = True

    USE_TENSORBOARD = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = ''
    RUN_NAME = 'default'
    LOSS = 'cross_entropy'
    OPTIMIZER = 'SGD'

    def __str__(self):
        fields = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        s = 'CONFIG PARAMETERS'
        for f in fields:
            s += f+': '+str(getattr(self, f))+'\n'
        return s


# class CONFIG_CIFAR10(DefaultConfig):
#     LR = 0.001
#     L1_REG = 0
#     EPOCHS = 5
#
# class KAF_CONFIG_CIFAR10(DefaultConfig):
#     LR = 0.001
#     L1_REG = 1e-4
#     EPOCHS = 5
