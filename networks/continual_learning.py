import quadprog
import torch.nn as nn
from copy import deepcopy
from networks.net_utils import AbstractNetwork
from utils.datasetsUtils.dataset import GeneralDatasetLoader
import torch.nn.functional as F
import random
from torch import stack, cat, mm, Tensor, clamp, zeros_like, from_numpy, unsqueeze, matmul, mul, abs
import numpy as np
from scipy.stats import multivariate_normal


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
        penality = self._penalty(kwargs['model'])
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

    def _penalty(self, model: AbstractNetwork):
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
        penality = self._penalty()
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

    def _penalty(self):
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
        self.margin = config.EWC_IMPORTANCE

        self.current_gradients = {}
        self.tasks_gradients = {}

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
        P = 0.5 * (P + P.transpose())  # + np.eye(t) * eps
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
                # print(a)
                # del a
                cg_np = cg.unsqueeze(1).cpu().contiguous().numpy().astype(np.double)#.astype(np.float16)
                tg = tg.numpy().transpose().astype(np.double)#.astype(np.float16)

                v = self._qp(tg, cg_np)

                cg_np += np.expand_dims(np.dot(v, tg), 1)

                del tg

                for name, p in self.model.named_parameters():
                    if name == n:
                        p.grad.data.copy_(from_numpy(cg_np).view(p.size()))

        return done

    def _constraints_gradients1(self, current_task, current_gradients):

        cg = []
        for n, g in current_gradients.items():
            cg.append(g)
        cg = cat(cg, 0)

        tg = []
        for t, tgs in self.tasks_gradients.items():
            if t >= current_task:
                continue
            ctg = []
            for n, g in tgs.items():
                ctg.append(g)
            ctg = cat(ctg, 0)
            tg.append(ctg)

        tg = stack(tg, 1)
        a = mm(cg.unsqueeze(0), tg)

        if (a < 0).sum() != 0:
            cg_np = cg.unsqueeze(1).cpu().contiguous().numpy().astype(np.double)#.astype(np.float16)
            del cg

            tg_np = tg.cpu().numpy().transpose().astype(np.double)#.astype(np.float16)

            v = self._qp(tg_np, cg_np)

            cg_np += np.expand_dims(np.dot(v, tg_np), 1)
            del tg, tg_np

            i = 0

            for name, p in self.model.named_parameters():
                size = p.size()
                flat_size = np.prod(size)#.view(-1)
                p.grad.data.copy_(Tensor(cg_np[i: i+flat_size]).view(size))
                i += flat_size

            # cg = []
            # for n, g in self.model.named_parameters():
            #     if g.requires_grad:
            #         cg.append(deepcopy(g.grad.data.view(-1)))
            # cg = cat(cg, 0)
            #
            # tg = []
            # for t, tgs in self.tasks_gradients.items():
            #     if t >= current_task:
            #         continue
            #     ctg = []
            #     for n, g in tgs.items():
            #         ctg.append(g)
            #     ctg = cat(ctg, 0)
            #     tg.append(ctg)
            #
            # tg = stack(tg, 1)
            # print(a)
            # a = mm(cg.unsqueeze(0), tg)
            # print(a)
            # print()
            return True
        else:
            return False

    def _save_gradients(self, current_task):

        if current_task == 0:
            return

        self._current_task = current_task
        self.tasks.append(current_task-1)

        self.model.eval()

        for sub_task in range(current_task):

            self.model.task = sub_task
            self.model.zero_grad()

            for images, label in self.dataset.getIterator(10, task=sub_task):
                # it = self.dataset.getIterator(32, task=sub_task)
                # images, _ = next(it)

                input = images.to(self.device)
                label = label.to(self.device)

                if self.is_conv:
                    output = self.model(input.view(1, input.shape[0], input.shape[1], input.shape[2]))
                else:
                    output = self.model(input)#.view(1, -1))

                # label = output.max(1)[1].view(-1)

                loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                loss.backward()
                break

            gradients = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    gradients[n] = p.grad.data.view(-1)

            self.tasks_gradients[sub_task] = deepcopy(gradients)

