from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F

from configs.configs import DefaultConfig
from utils.datasetsUtils.dataset import GeneralDatasetLoader


class EWC(object):
    # An utility object for computing the EWC penalty

    def __init__(self, model: nn.Module, dataset: GeneralDatasetLoader, config: DefaultConfig):
        # model: the model (either initialized or pretrained on previous tasks)
        # dataset: a list of (x, task_idx) elements from the previous tasks
        # device: current device in use
        # config: a configuration dictionary (see /configs/)

        self.model = model

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE

        # Save trainable parameters of the model in a dictionary
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        # Get means and precisions
        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = p.to(self.device)

        self._precision_matrices = None

    def __call__(self, *args, **kwargs):
        self._diag_fisher(old_tasks=kwargs['old_tasks'])
        return self

    def _diag_fisher(self, old_tasks):
        # Initialize precision matrices
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()

        for (input, task_idx) in old_tasks:

            self.model.zero_grad()
            self.model.set_task(task_idx)
            input = input.to(self.device)

            if self.is_conv:
                output = self.model(input.view(1, input.shape[0], input.shape[1], input.shape[2]))
            else:
                output = self.model(input.view(1, -1))

            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(old_tasks)

        precision_matrices = {n: p for n, p in precision_matrices.items()}

        self._precision_matrices = precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        if self._precision_matrices is None:
            return 0

        for n, p in model.named_parameters():

            # print(self._precision_matrices[n])
            # print(self._means[n])

            _loss = self._precision_matrices[n] * (p.to(self.device) - self._means[n]) ** 2
            loss += _loss.sum()

        return loss
