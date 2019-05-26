import quadprog
from copy import deepcopy
from networks.net_utils import AbstractNetwork
from utils.datasetsUtils.dataset import GeneralDatasetLoader
import torch.nn.functional as F
import random
from torch import stack, cat, mm, Tensor, clamp, zeros_like, from_numpy, unsqueeze, matmul, mul, abs, nn
import numpy as np
from scipy.stats import multivariate_normal


class EWC(object):

    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset
        self.is_incremental = config.IS_INCREMENTAL

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE

        self.sample_size = config.CL_PAR.get('sample_size', 200)

        self.importance = config.CL_PAR.get('penalty_importance', 1e3)

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._current_task = 0
        self._precision_matrices = {}

    def __call__(self, *args, **kwargs):
        if self.importance == 0:
            return self, 0
        self._update_matrix(kwargs['current_task'])
        penality = self._penalty()
        penality *= self.importance
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
            # self.model.task = task_idx

            if self.is_incremental:
                self.model.task = self.dataset.task_mask(task_idx)
            else:
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

    def _penalty(self):
        loss = 0
        if len(self._precision_matrices) == 0:
            return 0

        for t in self._precision_matrices.keys():
            if t == self._current_task:
                continue

            fisher = self._precision_matrices[t]
            means = self._means[t]
            for n, p in self.model.named_parameters():
                if n not in self.params:
                    continue
                _loss = fisher[n] * (p.to(self.device) - means[n]) ** 2
                loss += _loss.sum()

        return loss


class OnlineEWC(object):

    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset
        self.is_incremental = config.IS_INCREMENTAL

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE

        self.sample_size = config.CL_PAR.get('sample_size', 200)
        self.gamma = config.CL_PAR.get('gamma', 1)
        self.maxf = config.CL_PAR.get('maxf', 0.001)
        self.importance = config.CL_PAR.get('penalty_importance', (1/(config.LR*self.maxf))/2)

        self._current_task = 0
        self._batch_counter = 0

        self.tasks = []
        self._means = None
        self._precision_matrices = None
        self.loss_function = nn.CrossEntropyLoss()

    def __call__(self, *args, **kwargs):
        if self.importance == 0:
            return self, 0

        self._update_matrix(kwargs['current_task'])

        penalty = self._penalty()
        penalty *= self.importance
        return self, penalty

    def _update_matrix(self, current_task):

        if current_task-1 in self.tasks or current_task == 0:
            return

        self._batch_counter += 1
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

        self.dataset.train_phase()
        it = self.dataset.getIterator(self.sample_size, task=current_task-1)
        images, labels = next(it)
        old_tasks.extend([(images[i], labels[i], current_task-1) for i in range(len(images))])
        old_tasks = random.sample(old_tasks, k=self.sample_size)

        for (input, label, task_idx) in old_tasks:

            self.model.zero_grad()
            # self.model.task = task_idx

            # if self.is_incremental:
            #     self.model.task = self.dataset.task_mask(task_idx)
            # else:
            #     self.model.task = task_idx

            input = input.to(self.device)
            label = label.unsqueeze(0).to(self.device)

            if self.is_conv:
                output = self.model(input.view(1, input.shape[0], input.shape[1], input.shape[2]))
            else:
                output = self.model(input.view(1, -1))

            label = output.max(1)[1].view(-1)
            loss = self.loss_function(output, label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += (p.grad.data ** 2) / len(old_tasks)

        if self._precision_matrices is not None:
            for n, p in precision_matrices.items():
                self._precision_matrices[n].data += self.gamma * precision_matrices[n].data
                self._precision_matrices[n].data = clamp(self._precision_matrices[n].data/self._batch_counter,
                                                         max=self.maxf)
        else:
            self._precision_matrices = precision_matrices

        self.model.train()

    def _penalty(self):
        loss = 0

        if self._precision_matrices is None:
            return 0

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                _loss = self._precision_matrices[n] * \
                        (p - self._means[n]) ** 2
                loss += _loss.sum()

        return loss.to(self.device)


class GEM(object):
    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset
        self.is_incremental = config.IS_INCREMENTAL

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE
        self._current_task = 0
        self.margin = config.CL_PAR.get('margin', 0.5)

        self.current_gradients = {}
        self.tasks_gradients = {}
        self.sample_size = config.CL_PAR.get('sample_size', 100)

        self.tasks = []
        self._precision_matrices = None

    def __call__(self, *args, **kwargs):

        if kwargs['current_task'] == 0:
            return self, 0

        current_gradients = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                current_gradients[n] = deepcopy(p.grad.data.view(-1))

        self._save_gradients(current_task=kwargs['current_task'])

        done = self._constraints_gradients(current_task=kwargs['current_task'], current_gradients=current_gradients)

        if not done:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.grad.copy_(current_gradients[n].view(p.grad.data.size()))

        return self, 0

    def _qp(self, past_tasks_gradient, current_gradient):
        t = past_tasks_gradient.shape[0]
        P = np.dot(past_tasks_gradient, past_tasks_gradient.transpose())
        P = 0.5 * (P + P.transpose()) # + np.eye(t) * eps
        q = np.dot(past_tasks_gradient, current_gradient) * -1
        q = np.squeeze(q, 1)
        h = np.zeros(t) + self.margin
        G = np.eye(t)
        v = quadprog.solve_qp(P, q, G, h)[0]
        return v

    def _constraints_gradients(self, current_task, current_gradients):

        done = False

        for n, cg in current_gradients.items():
            tg = []
            for t, tgs in self.tasks_gradients.items():
                if t < current_task:
                    tg.append(tgs[n])

            tg = stack(tg, 1).cpu()
            a = mm(cg.unsqueeze(0).cpu(), tg)

            if (a < 0).sum() != 0:
                done = True
                cg_np = cg.unsqueeze(1).cpu().contiguous().numpy().astype(np.double)#.astype(np.float16)
                tg = tg.numpy().transpose().astype(np.double)#.astype(np.float16)

                v = self._qp(tg, cg_np)

                cg_np += np.expand_dims(np.dot(v, tg), 1)

                del tg

                for name, p in self.model.named_parameters():
                    if name == n:
                        p.grad.data.copy_(from_numpy(cg_np).view(p.size()))

        return done

    def _save_gradients(self, current_task):

        if current_task == 0:
            return

        self._current_task = current_task
        self.tasks.append(current_task-1)

        self.model.eval()

        for sub_task in range(current_task):

            # self.model.task = sub_task
            # if self.is_incremental:
            #     self.model.task = self.dataset.task_mask(sub_task)
            # else:
            self.model.task = sub_task

            self.model.zero_grad()

            for images, label in self.dataset.getIterator(self.sample_size, task=sub_task):

                input = images.to(self.device)
                label = label.to(self.device)

                # if self.is_conv:
                #     output = self.model(input.view(1, input.shape[0], input.shape[1], input.shape[2]))
                # else:
                output = self.model(input)

                loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                loss.backward()
                break

            gradients = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    gradients[n] = p.grad.data.view(-1)

            self.tasks_gradients[sub_task] = deepcopy(gradients)

