
import torch

config = {

    'task': {

        'num_tasks': 10,
        'is_conv': True,

    },

    'ewc': {

        'importance': 1000,
        'sample_size': 250

    },

    'opt': {

        'lr': 1e-3,
        'l1_reg': 0,
        'iters': 1,
        'batch_size': 64,
        'epochs': 15
    },

    'other': {

        'enable_tensorboard': True,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'model_name': '', # To be modified dinamically
        'run_name': 'cifar10' # To be modified dinamically

    },

}