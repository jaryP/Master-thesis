import quadprog
import torch.nn as nn
from copy import deepcopy
from networks.net_utils import AbstractNetwork
from utils.datasetsUtils.dataset import GeneralDatasetLoader
import torch.nn.functional as F
import random
from torch import stack, cat, mm, Tensor, clamp, zeros_like, from_numpy, unsqueeze, matmul, mul, abs, div
import numpy as np
from scipy.stats import multivariate_normal


class Bayesian(object):
    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE
        self.margin = config.EWC_IMPORTANCE

        self.sample_size_task = 0
        self._current_task = 0

        self.current_gradients = {}
        self.tasks_gradients = {}

        self.tasks = []
        self._precision_matrices = {}

    def __call__(self, *args, **kwargs):

        self._save_parameters(current_task=kwargs['current_task'])
        penalty = self._penalty()
        return self, penalty

    def _save_parameters(self, current_task):

        if current_task not in self._precision_matrices:
            self.sample_size_task = 0
            mean = {}
            variance = {}
            self._current_task = current_task

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    mean[n] = deepcopy(p).to('cpu')
                    variance[n] = deepcopy(p).to('cpu') ** 2

            self._precision_matrices[current_task] = {'mean': mean, 'variance': variance, 'w': None}
            if current_task > 0:
                mean = {}
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        mean[n] = deepcopy(p).to('cpu')

                self._precision_matrices[current_task - 1].update({'w': mean})

        self.sample_size_task += 1

        if self.sample_size_task % self.sample_size == 0:
            m = self.sample_size_task // self.sample_size
            mean = self._precision_matrices[current_task]['mean']
            # variance = self._precision_matrices[current_task]['variance']

            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    mean[n] = (m * mean[n] + p.to('cpu')) / (m + 1)
                    # variance[n] = (m*variance[n] + p.to('cpu')**2) / (m+1)

            self._precision_matrices[current_task]['mean'] = mean
            # self._precision_matrices[current_task]['variance'] = variance

    def _penalty(self):
        loss = 0

        if len(self._precision_matrices) <= 1:
            return 0

        for t in self._precision_matrices.keys():
            if t != self._current_task:
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        mean = self._precision_matrices[t]['mean'][n]
                        # variance = self._precision_matrices[t]['variance'][n]
                        #
                        # diag = variance - (mean ** 2)

                        _loss = (p.to('cpu') - mean) ** 2
                        loss += _loss.sum()

        return loss

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
            cg_np = cg.unsqueeze(1).cpu().contiguous().numpy().astype(np.double)  # .astype(np.float16)
            del cg

            tg_np = tg.cpu().numpy().transpose().astype(np.double)  # .astype(np.float16)

            v = self._qp(tg_np, cg_np)

            cg_np += np.expand_dims(np.dot(v, tg_np), 1)
            del tg, tg_np

            i = 0

            for name, p in self.model.named_parameters():
                size = p.size()
                flat_size = np.prod(size)  # .view(-1)
                p.grad.data.copy_(Tensor(cg_np[i: i + flat_size]).view(size))
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
        self.tasks.append(current_task - 1)

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
                    output = self.model(input)  # .view(1, -1))

                # label = output.max(1)[1].view(-1)

                loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                loss.backward()
                break

            gradients = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    gradients[n] = p.grad.data.view(-1)

            self.tasks_gradients[sub_task] = deepcopy(gradients)


class SI(object):
    # An utility object for computing the EWC penalty

    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset
        self.w0 = 0.00001
        self.w = 0.005
        self.eps = 1e-7

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self._current_task = 0

        self.tasks = set()

        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._last_means = {n: p.to(self.device) for n, p in deepcopy(params).items()}

        self._means = {}

        self.first_batch = True
        self._precision_matrices = {}

    def __call__(self, *args, **kwargs):
        self._update_matrix(kwargs['current_task'])

        penality = self._penalty()

        return self, penality

    def _update_matrix(self, current_task):

        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        grads = {n: deepcopy(p.grad) for n, p in self.model.named_parameters() if p.requires_grad}

        if current_task > 0 and current_task - 1 not in self.tasks:
            self._means[current_task - 1] = {n: p.to(self.device) for n, p in deepcopy(params).items()}
            # self._means = {n: p.to(self.device) for n, p in deepcopy(params).items()}

        if current_task not in self._precision_matrices:
            self._precision_matrices[current_task] = {n: zeros_like(p) for n, p in deepcopy(params).items()}

        if current_task - 1 not in self._means:
            T = {n: self.eps
                 for n, p in self.model.named_parameters() if p.requires_grad}
        else:
            T = {n: (p - self._means[current_task - 1][n]) ** 2 + self.eps
                 for n, p in self.model.named_parameters() if p.requires_grad}

        self._current_task = current_task
        self.tasks.add(current_task - 1)

        if self.first_batch:
            w = self.w0
            self.first_batch = False
        else:
            w = self.w

        for n, p in grads.items():
            L = self._last_means[n].data - params[n].data
            L = L * p.data

            v = L / T[n]
            v = F.relu(v)
            v = self._precision_matrices[current_task][n].data + w * v
            v = clamp(v, max=2)
            self._precision_matrices[current_task][n].data = v

        self._last_means = {n: p.to(self.device) for n, p in deepcopy(params).items()}

    def _penalty(self):
        loss = 0

        if self._current_task == 0:
            return 0

        for task in self._precision_matrices.keys():
            if task >= self._current_task:
                continue
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    # print(self._precision_matrices[task][n])
                    _loss = self._precision_matrices[task][n] * (p.to(self.device) - self._means[task][n]) ** 2
                    loss += _loss.sum()

        return loss


class Jary(object):
    # An utility object for computing the EWC penalty

    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset
        self.eps = 1e-7

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self._current_task = 0
        self.sample_size_task = 0

        self.tasks = set()

        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._last_means = {n: p.to(self.device) for n, p in deepcopy(params).items()}

        self._means = {}

        self.first_batch = True
        self._precision_matrices = {}
        self._normals = {}

    def __call__(self, *args, **kwargs):
        self._update_matrix(kwargs['current_task'])

        penality = self._penalty()

        return self, penality

    def _update_matrix(self, current_task):

        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        grads = {n: deepcopy(p.grad) for n, p in self.model.named_parameters() if p.requires_grad}

        if current_task > 0 and current_task - 1 not in self._normals:
            self._means[current_task - 1] = {n: p.to(self.device) for n, p in deepcopy(params).items()}

            tot_d = {}
            for n, p in self._precision_matrices.items():
                mean = p['mean'].detach().cpu().numpy()
                mean = np.squeeze(mean)

                variance = p['variance'].detach().cpu().numpy()
                variance = np.squeeze(variance)

                d = {}
                if len(mean.shape) == 2:
                    for i in range(mean.shape[0]):
                        cov = variance[i]
                        # cov[cov == 0] = 1
                        # cov = np.diag(cov) + np.eye(len(cov))
                        # a = np.random.multivariate_normal(mean[i], cov)
                        d = {'mean': mean[i], 'cov': cov}
                else:
                    cov = variance
                    # cov[cov == 0] = 1
                    # cov = np.diag(cov) + np.eye(len(cov))
                    d = {'mean': mean, 'cov': cov}

                tot_d[n] = d

            self._normals[current_task - 1] = tot_d

            self._precision_matrices = {}

            self.sample_size_task = 0

        if self.sample_size_task % self.sample_size == 0:
            m = self.sample_size_task // self.sample_size

            if len(self._precision_matrices) == 0:
                self._precision_matrices = {n: {'mean': zeros_like(p), 'variance': zeros_like(p)}
                                            for n, p in deepcopy(params).items()}

            # if current_task - 1 not in self._means:
            #     T = {n: self.eps
            #          for n, p in self.model.named_parameters() if p.requires_grad}
            # else:
            #     T = {n: (p-self._means[current_task-1][n])**2 + self.eps
            #          for n, p in self.model.named_parameters() if p.requires_grad}

            self._current_task = current_task
            self.tasks.add(current_task - 1)

            for n, p in grads.items():
                L = self._last_means[n].data - params[n].data
                L = L * p.data

                # self._precision_matrices[current_task][n]['mean'].data =

                mean = self._precision_matrices[n]['mean']
                variance = self._precision_matrices[n]['variance']

                # for n, p in self.model.named_parameters():
                #     if p.requires_grad:
                mean = (m * mean + L) / (m + 1)
                variance = (m * variance + L ** 2) / (m + 1)

                # v = L/T[n]
                # v = F.relu(v)
                # v = self._precision_matrices[current_task][n].data + w*v
                # v = clamp(v, max=2)
                # self._precision_matrices[current_task][n].data = v
                self._precision_matrices[n]['mean'] = mean
                self._precision_matrices[n]['variance'] = variance

        self._last_means = {n: p.to(self.device) for n, p in deepcopy(params).items()}

    def _penalty(self):
        loss = 0

        if self._current_task == 0:
            return 0

        for task in self._normals.keys():
            if task >= self._current_task:
                continue
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    toUnsq = False
                    # print(self._precision_matrices[task][n])
                    size = np.array(p.size())

                    if size[0] == 1:
                        toUnsq = True
                        size = size[1:]  # np.squeeze(size, 0)

                    normal = self._normals[task][n]
                    mean, cov = normal['mean'], normal['cov']

                    if len(size) > 1:
                        m = []
                        for i in range(size[0]):
                            mcom = np.random.normal(mean, cov)
                            m.append(from_numpy(mcom))
                        m = stack(m, 0)
                    else:
                        mcom = np.random.normal(mean, cov)
                        m = from_numpy(np.squeeze(mcom))

                    m = m.float().to(self.device)
                    m = F.relu(m)

                    if toUnsq:
                        m = unsqueeze(m, 0)
                    # m2 = stack(m, 1)
                    # m3 = stack(m, 2)

                    _loss = (p.to(self.device) - self._means[task][n]) ** 2
                    _loss *= m
                    # _loss = self._precision_matrices[task][n] * (p.to(self.device) - self._means[task][n]) ** 2
                    loss += _loss.sum()
                    # print(loss)
                    # continue

        return loss


class JaryGEM(object):
    # An utility object for computing the EWC penalty

    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset
        self.eps = 1e-7

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self._current_task = 0
        self.sample_size_task = 0

        self.margin = config.EWC_IMPORTANCE

        self.tasks = set()

        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._last_means = {n: p.to(self.device) for n, p in deepcopy(params).items()}

        self._means = {}

        self.first_batch = True
        self._precision_matrices = {}
        self._normals = {}

    def __call__(self, *args, **kwargs):

        self._update_matrix(kwargs['current_task'])

        if kwargs['current_task'] == 0:
            return self, 0

        # current_gradients = {}
        # for n, p in self.model.named_parameters():
        #     if p.requires_grad:
        #         current_gradients[n] = deepcopy(p.grad.data.view(-1))

        self._constraints_gradients(current_task=kwargs['current_task'])

        # penality = self._penalty()

        self._last_means = {n: deepcopy(p) for n, p in self.model.named_parameters() if p.requires_grad}

        return self, 0

    def _constraints_gradients(self, current_task):

        done = False

        if current_task - 1 not in self._means:
            T = {n: self.eps
                 for n, p in self.model.named_parameters() if p.requires_grad}
        else:
            T = {n: (p - self._means[current_task - 1][n]).cpu() + self.eps
                 for n, p in self.model.named_parameters() if p.requires_grad}

        for n, cg in self.model.named_parameters():
            # ccg = cg.detach().view(-1).unsqueeze(0).cpu()

            g = cg.grad.data
            diff = cg.data - self._last_means[n].data

            L = diff
            L = mul(L, g).cpu()
            L = div(L, T[n])
            L = L.view(-1).unsqueeze(0).detach().cpu()
            # L = F.relu(L)

            tg = []
            for t, tgs in self._normals.items():

                if t < current_task:
                    normal = tgs[n]
                    mean, cov = normal['mean'], normal['cov']
                    mcom = np.random.normal(mean, cov)
                    mcom = from_numpy(mcom)

                    tg.append(mcom)

            tg = stack(tg, 1).float().cpu()
            a = mm(L, tg)
            print(a)

            if (a < 0).sum() != 0:
                done = True
                # print(a)
                # del a
                L = L.permute(1, 0).contiguous().numpy().astype(np.double)
                # cg = cg.detach().view(-1).unsqueeze(1).cpu().contiguous().numpy().astype(np.double)#.astype(np.float16)
                tg = tg.numpy().transpose().astype(np.double)  # .astype(np.float16)

                v = self._qp(tg, L)

                # cg += np.expand_dims(np.dot(v, tg), 1)
                tg = np.expand_dims(np.dot(v, tg), 1)
                tg = from_numpy(tg).to(self.device).float()
                tg = tg.view(cg.size())

                # tg = mul(tg, 1 / diff)

                ng = cg + tg
                # ng = mul(ng, 1 / diff)
                # print(v)

                # print(cg.device, ng.device)

                del tg

                # for name, p in self.model.named_parameters():
                #     if name == n:
                cg.grad.data.copy_(ng)
                # break

        return done

    def _update_matrix(self, current_task):

        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        grads = {n: deepcopy(p.grad) for n, p in self.model.named_parameters() if p.requires_grad}

        if current_task > 0 and current_task - 1 not in self._normals:
            self._means[current_task - 1] = {n: p.to(self.device) for n, p in deepcopy(params).items()}

            tot_d = {}
            for n, p in self._precision_matrices.items():
                mean = p['mean'].detach().cpu().numpy()
                mean = np.squeeze(mean).reshape(-1)

                variance = p['variance'].detach().cpu().numpy()
                variance = np.squeeze(variance).reshape(-1)

                tot_d[n] = {'mean': mean, 'cov': variance}

            self._normals[current_task - 1] = tot_d

            self._precision_matrices = {}

            self.sample_size_task = 0

        if self.sample_size_task % self.sample_size == 0:
            m = self.sample_size_task // self.sample_size

            if len(self._precision_matrices) == 0:
                self._precision_matrices = {n: {'mean': zeros_like(p).cpu(),
                                                'variance': zeros_like(p).cpu()}
                                            for n, p in deepcopy(params).items()}

            if current_task - 1 not in self._means:
                T = {n: self.eps
                     for n, p in self.model.named_parameters() if p.requires_grad}
            else:
                T = {n: ((p-self._means[current_task-1][n])**2).cpu() + self.eps
                     for n, p in self.model.named_parameters() if p.requires_grad}

            self._current_task = current_task
            self.tasks.add(current_task - 1)

            for n, p in grads.items():
                L = params[n].data - self._last_means[n].data
                L = L * p.data
                L = L.cpu()
                L = L/T[n]
                # L = F.relu(L)

                mean = self._precision_matrices[n]['mean']
                variance = self._precision_matrices[n]['variance']

                mean = (m * mean + L) / (m + 1)
                variance = (m * variance + L ** 2) / (m + 1)

                # v = F.relu(v)
                # v = self._precision_matrices[current_task][n].data + w*v
                # v = clamp(v, max=2)
                # self._precision_matrices[current_task][n].data = v

                self._precision_matrices[n]['mean'] = mean
                self._precision_matrices[n]['variance'] = variance

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

    def _penalty(self):
        loss = 0

        if self._current_task > 0:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    g = p.grad.data.view(-1)
                    L = p.data - self._last_means[n].data
                    L = L.view(-1) * g

                    normal = self._normals[0][n]
                    mean, cov = normal['mean'], normal['cov']
                    mcom = np.random.normal(mean, cov)
                    mcom = from_numpy(mcom)
                    a = matmul(L.cpu(), mcom.float())

        return loss

        if self._current_task == 0:
            return 0

        for task in self._normals.keys():
            if task >= self._current_task:
                continue
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    toUnsq = False
                    # print(self._precision_matrices[task][n])
                    size = np.array(p.size())

                    if size[0] == 1:
                        toUnsq = True
                        size = size[1:]  # np.squeeze(size, 0)

                    normal = self._normals[task][n]
                    mean, cov = normal['mean'], normal['cov']

                    if len(size) > 1:
                        m = []
                        for i in range(size[0]):
                            mcom = np.random.normal(mean, cov)
                            m.append(from_numpy(mcom))
                        m = stack(m, 0)
                    else:
                        mcom = np.random.normal(mean, cov)
                        m = from_numpy(np.squeeze(mcom))

                    m = m.float().to(self.device)
                    m = F.relu(m)

                    if toUnsq:
                        m = unsqueeze(m, 0)
                    # m2 = stack(m, 1)
                    # m3 = stack(m, 2)

                    _loss = (p.to(self.device) - self._means[task][n]) ** 2
                    _loss *= m
                    # _loss = self._precision_matrices[task][n] * (p.to(self.device) - self._means[task][n]) ** 2
                    loss += _loss.sum()
                    # print(loss)
                    # continue

        return loss
