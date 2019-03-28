import torch.nn as nn
from copy import deepcopy
from networks.net_utils import AbstractNetwork
# import configs.configClasses as configClasses
from utils.datasetsUtils.dataset import GeneralDatasetLoader
import torch.nn.functional as F
import random


class RealEWC(object):
    # An utility object for computing the EWC penalty

    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._current_task = 0
        self._precision_matrices = {}

    def __call__(self, *args, **kwargs):
        self._update_matrix(kwargs['current_task'])
        penality = self._penality(kwargs['model'])
        return self, penality

    def _update_matrix(self, current_task):

        self._current_task = current_task

        if current_task == 0:
            return

        if current_task-1 not in self._means:
            self._means[current_task-1] = {n: p.to(self.device) for n, p in deepcopy(self.params).items()}

        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()
        self.dataset.train_phase()

        old_tasks = []
        for sub_task in range(current_task):
            it = self.dataset.getIterator(self.sample_size, task=sub_task)
            images, _ = next(it)
            old_tasks.extend([(images[i], sub_task) for i in range(len(images))])

        old_tasks = random.sample(old_tasks, k=self.sample_size)

        for (input, task_idx) in old_tasks:

            self.model.zero_grad()
            self.model.task = task_idx
            input = input.to(self.device)

            if self.is_conv:
                output = self.model(input.view(1, input.shape[0], input.shape[1], input.shape[2]))
            else:
                output = self.model(input.view(1, -1))

            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if n not in self.params:
                    continue
                precision_matrices[n].data += (p.grad.data ** 2) / len(old_tasks)

            self._precision_matrices[task_idx] = precision_matrices

    def _penality(self, model: AbstractNetwork):
        loss = 0
        if len(self._precision_matrices) == 0:
            return 0

        for t in self._precision_matrices.keys():
            if t == self._current_task:
                continue

            fisher = self._precision_matrices[t]
            means = self._means[t]
            for n, p in model.named_parameters():
                if n not in self.params:
                    continue
                _loss = fisher[n] * (p.to(self.device) - means[n]) ** 2
                loss += _loss.sum()

        return loss


class EWC(object):
    # An utility object for computing the EWC penalty

    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE

        # self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.params = None
        self._means = None
        # self._means = {}
        # for n, p in deepcopy(self.params).items():
        #     self._means[n] = p.to(self.device)

        self._precision_matrices = None

    def __call__(self, *args, **kwargs):
        self._update_matrix(kwargs['current_task'])
        return self

    def _update_matrix(self, current_task):

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        self._means = {}
        for n, p in deepcopy(self.params).items():
            self._means[n] = p.to(self.device)

        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()

        old_tasks = []
        for sub_task in range(current_task):
            self.dataset.train_phase()
            it = self.dataset.getIterator(self.sample_size, task=sub_task)

            images, _ = next(it)

            old_tasks.extend([(images[i], sub_task) for i in range(len(images))])

        old_tasks = random.sample(old_tasks, k=self.sample_size)

        for (input, task_idx) in old_tasks:

            self.model.zero_grad()
            self.model.task = task_idx
            input = input.to(self.device)

            if self.is_conv:
                output = self.model(input.view(1, input.shape[0], input.shape[1], input.shape[2]))
            else:
                output = self.model(input.view(1, -1))

            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += (p.grad.data ** 2) / len(old_tasks)

        precision_matrices = {n: p for n, p in precision_matrices.items()}

        self._precision_matrices = precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0

        if self._precision_matrices is None:
            return 0

        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p.to(self.device) - self._means[n]) ** 2
            loss += _loss.sum()

        return loss


class OnlineEWC(object):
    # An utility object for computing the EWC penalty

    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE
        self.gamma = config.GAMMA
        self._current_task = 0

        self.tasks = []
        self._means = None
        self._precision_matrices = None

    def __call__(self, *args, **kwargs):
        self._update_matrix(kwargs['current_task'])
        penality = self._penality()
        return self, penality

    def _update_matrix(self, current_task):

        if current_task-1 in self.tasks or current_task == 0:
            return

        self._current_task = current_task
        self.tasks.append(current_task-1)

        # self._means = {n: p.to(self.device) for n, p in deepcopy(self.params).items()}
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = deepcopy(params)

        self.model.eval()

        precision_matrices = {}
        for n, p in deepcopy(params).items():
            if p.requires_grad:
                p.data.zero_()
                precision_matrices[n] = p.data

        old_tasks = []

        # for sub_task in range(current_task):
        self.dataset.train_phase()
        it = self.dataset.getIterator(self.sample_size, task=current_task-1)
        images, _ = next(it)
        old_tasks.extend([(images[i], current_task-1) for i in range(len(images))])
        old_tasks = random.sample(old_tasks, k=self.sample_size)

        for (input, task_idx) in old_tasks:

            self.model.zero_grad()
            self.model.task = task_idx
            input = input.to(self.device)

            if self.is_conv:
                output = self.model(input.view(1, input.shape[0], input.shape[1], input.shape[2]))
            else:
                output = self.model(input.view(1, -1))

            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += (p.grad.data ** 2) / len(old_tasks)

        if self._precision_matrices is not None:
            for n, p in precision_matrices.items():
                self._precision_matrices[n].data += self.gamma * precision_matrices[n].data
        else:
            self._precision_matrices = precision_matrices

    def _penality(self):
        loss = 0

        if self._precision_matrices is None:
            return 0

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * (p.to(self.device) - self._means[n]) ** 2
                loss += _loss.sum()

        return loss


class GEM(object):
    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE
        self._current_task = 0

        self.current_gradients = {}
        self.tasks_gradients = {}

        self.tasks = []
        self._precision_matrices = None

    def __call__(self, *args, **kwargs):

        current_gradients = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                current_gradients[n] = deepcopy(p.grad.data.view(-1))

        self._save_gradients(current_task=kwargs['current_task'])

        self._constraints_gradients(current_task=kwargs['current_task'], current_gradients=current_gradients)

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.copy_(current_gradients[n])

        return self, 0

    def _constraints_gradients(self, current_task, current_gradients):

        violated = False
        for t, g in self.tasks_gradients:
            if t < current_task:
                dot = 0
                if dot < 0:
                    violated=True
                    break

        if violated:
            pass

    def _save_gradients(self, current_task):

        if current_task-1 in self.tasks or current_task == 0:
            return

        self._current_task = current_task
        self.tasks.append(current_task-1)

        self.model.eval()

        old_tasks = []
        for sub_task in range(current_task):
            it = self.dataset.getIterator(self.sample_size, task=sub_task)
            images, _ = next(it)
            old_tasks.extend([(images[i], sub_task) for i in range(len(images))])
        old_tasks = random.sample(old_tasks, k=self.sample_size)

        for (input, task_idx) in old_tasks:

            self.model.zero_grad()
            self.model.task = task_idx
            input = input.to(self.device)

            if self.is_conv:
                output = self.model(input.view(1, input.shape[0], input.shape[1], input.shape[2]))
            else:
                output = self.model(input.view(1, -1))

            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            gradients = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    gradients[n] = p.grad.data.view(-1)

            self.tasks_gradients[task_idx] = gradients
