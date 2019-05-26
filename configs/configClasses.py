import torch
import networks.continual_learning as continual_learning
from networks.continual_learning_beta import embedding


class DefaultConfig(object):
    LR = 0.001
    L1_REG = 0
    IS_INCREMENTAL = False

    ITERS = 1
    EPOCHS = 2
    BATCH_SIZE = 64
    IS_CONVOLUTIONAL = True

    NEXT_TASK_LR = LR
    NEXT_TASK_EPOCHS = EPOCHS

    EWC_SAMPLE_SIZE = 250
    EWC_IMPORTANCE = 1000
    USE_CL = True

    CL_TEC = continual_learning.EWC
    CL_PAR = {'sample_size': 250, 'penalty_importance': 1e+3}

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
    CL_TEC = continual_learning.OnlineEWC
    GAMMA = 1.0
    CL_PAR = {'sample_size': 250, 'penalty_importance': 1e+3, 'gamma': 1}


class RealEwc(DefaultConfig):
    CL_TEC = continual_learning.EWC


class GEM(DefaultConfig):
    CL_TEC = continual_learning.GEM
    EWC_IMPORTANCE = 0.5
    CL_PAR = {'margin': 0.5}
    # super.CL_TEC_PARAMETERS['margin'] = 0.5


class Embedding(DefaultConfig):
    CL_TEC = embedding
    # CL_PAR = {'margin': 0.5}
    # super.CL_TEC_PARAMETERS['margin'] = 0.5

