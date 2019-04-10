import torch
import networks.continual_learning as continual_learning


class DefaultConfig(object):
    LR = 0.001
    L1_REG = 0

    ITERS = 1
    EPOCHS = 2
    BATCH_SIZE = 64
    IS_CONVOLUTIONAL = True

    EWC_SAMPLE_SIZE = 250
    EWC_IMPORTANCE = 1000
    USE_EWC = True
    EWC_TYPE = continual_learning.RealEWC

    USE_TENSORBOARD = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = ''
    SAVE_PATH = '.'

    RUN_NAME = 'default'
    LOSS = 'cross_entropy'
    OPTIMIZER = 'SGD'

    def __str__(self):
        fields = [a for a in dir(self) if not a.startswith('__')]
        s = 'CONFIG PARAMETERS\n'
        for f in fields:
            s += f+': '+str(getattr(self, f))+'\n'
        return s


class OnlineLearningConfig(DefaultConfig):
    EWC_TYPE = continual_learning.OnlineEWC
    GAMMA = 1.0


class RealEwc(DefaultConfig):
    EWC_TYPE = continual_learning.RealEWC


class GEM(DefaultConfig):
    EWC_TYPE = continual_learning.GEM
    EWC_IMPORTANCE = 0.5

