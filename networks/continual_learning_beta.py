import quadprog
import torch
import torch.nn as nn
from copy import deepcopy
from networks.net_utils import AbstractNetwork
from utils.datasetsUtils.dataset import GeneralDatasetLoader
import torch.nn.functional as F
import random
from torch import stack, cat, mm, Tensor, clamp, zeros_like, from_numpy, unsqueeze, matmul, mul, abs, div, optim
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.spatial.distance import euclidean, pdist, cdist
import matplotlib.pyplot as plt
from torch.autograd import grad


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


class embedding(object):
    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28*28, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 100),
            # torch.nn.ReLU(),
            # torch.nn.linear(100 * 28, 200),
        ).to(config.DEVICE)

        def hook(module, input, output):
            setattr(module, "_value_hook", output)

        for n, m in self.model.named_modules():
            if n != 'proj':
                m.register_forward_hook(hook)

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.sample_size = config.EWC_SAMPLE_SIZE
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self._current_task = 0
        self.sample_size_task = 0

        # self.margin = config.EWC_IMPORTANCE

        self.tasks = set()

        self.embeddings = None
        self.embeddings_images = None
        self.w = []
        self.embeddings_images = []

    def __call__(self, *args, **kwargs):
        current_task = kwargs['current_task']
        current_batch = kwargs['batch']
        penalty = 0
        # self.train_step(current_batch)
        # self.embedding_save(current_task)
        if current_task > 0:
            self.embedding_save(current_task)
            self.penalty_1()
        #     penalty = self.penalty(current_task, current_batch)
        return self, penalty

    def embedding_save(self, current_task):
        if current_task-1 not in self.tasks:
            self.tasks.add(current_task-1)

            self.model.eval()
            self.dataset.train_phase()

            it = self.dataset.getIterator(self.sample_size, task=current_task-1)
            # labels = [current_task-1] * self.sample_size
            # self.embeddings_labels.extend(labels)
            self.w.extend([1] * self.sample_size)

            images, _ = next(it)
            # for (images, _) in it:

            input = images.to(self.device)
            with torch.no_grad():
                output = self.model.embedding(input)

                embeddings = output.cpu()
                if self.embeddings is None:
                    self.embeddings = embeddings
                    self.embeddings_images = images
                    self.w = [1] * self.sample_size
                else:
                    self.embeddings = torch.cat((self.embeddings, embeddings), 0)
                    self.embeddings_images = torch.cat((self.embeddings_images, images), 0)
                    self.w = [1] * self.embeddings.size()[0]

    def train_step(self, current_batch):
        x = current_batch[0]
        y = self.decoder(self.encoder(x))
        self.optimizer.zero_grad()
        loss = self.loss(y, x)
        loss.backward()
        self.optimizer.step()

    def penalty_1(self):
        self.model.eval()
        idx = range(self.embeddings_images.size()[0])
        idx = random.choices(idx, k=self.batch_size, weights=self.w)

        img = self.embeddings_images[idx].to(self.device)
        embeddings = self.embeddings[idx].to(self.device)

        new_embeddings = self.model.embedding(img)

        # d = torch.bmm(embeddings.unsqueeze(1), new_embeddings.unsqueeze(-1))

        # x = embeddings / embeddings.norm(dim=1)[:, None]
        # y = new_embeddings / new_embeddings.norm(dim=1)[:, None]
        # dist = torch.mm(x, y.transpose(0, 1))
        cosine = torch.nn.functional.cosine_similarity(embeddings, new_embeddings)
        dist = 1-cosine
        # print(cosine)
        # print(dist)

        dist.mean().backward()

        dist = dist.detach().cpu().numpy()
        # print(dist)
        # print(self.w)
        for j, i in enumerate(idx):
            self.w[i] = dist[j]
        # print(self.w)
        # input()
        # print(dist.size())
        # print(dist.min())

    def penalty(self, current_task, current_batch):

        self.model.eval()

        self.dataset.train_phase()
        # it = self.dataset.getIterator(self.batch_size, task=current_task)
        #
        # images, _ = next(it)
        # # for (images, _) in it:
        #
        # i = images.to(self.device)
        #
        # if self.is_conv:
        #     i = self.encoder(i)
        # else:
        #     i = self.encoder(i)

        # embeddings = i.cpu().detach().numpy()
        #
        # if self.embeddings is None:
        #     self.embeddings = embeddings
        # else:
        #     self.embeddings = np.concatenate((self.embeddings, embeddings), 0)

        old_tasks = []
        tot_loss = 0
        self.dataset.train_phase()

        for sub_task in range(current_task):
            it = self.dataset.getIterator(self.batch_size, task=sub_task)

            images, _ = next(it)
            images = images.to(self.device)

            # old_tasks.extend([images[i] for i in range(len(images))])

            # images = torch.stack(random.sample(old_tasks, k=self.sample_size), 0).to(self.device)

            # with torch.no_grad():
            #     embeddings = self.encoder(images).cpu().numpy()
            #     W = np.exp(-(euclidean_distances(embeddings, embeddings)))
            # print(W)
            forward = self.model.embedding(images)
            forward_b = self.model.embedding(current_batch[0])

                # tot_p = []
                # for n, p in self.model.named_modules():
                #     if hasattr(p, '_value_hook'):
                #         # print(n, getattr(p, '_value_hook'))
                #         forward = getattr(p, '_value_hook').cpu().numpy()
                #
                #         X = np.matmul(forward, forward.T)
                #         m = np.diag(X)
                #         m = np.matmul(np.expand_dims(m, -1), np.ones((1, len(forward))))
                #         d = m + m.T - 2 * X
                #         d *= W
                #
                #         p = np.max(d, 1)
                #         p = np.mean(p)
                #         tot_p.append(p)
            # dis = torch.nn.PairwiseDistance()
            # d = dis(forward, forward)

            # EUCLIDEAN DISTANCEs
            # x_norm = (forward_b**2).sum(1).view(-1, 1)
            # y_norm = (forward**2).sum(1).view(1, -1)
            # dist = x_norm + y_norm - 2.0 * torch.mm(forward_b, torch.transpose(forward, 0, 1))

            # COSINE SIMILARITY
            x = forward_b / forward_b.norm(dim=1)[:, None]
            y = forward / forward.norm(dim=1)[:, None]
            dist = torch.mm(x, y.transpose(0, 1))
            # print(torch.sum((dist < 0).int()))
            # dist = torch.mm(forward_b, torch.transpose(forward, 0, 1))
            # dist = torch.mm(forward_b.t(), forward)
            # dist /= x_norm / y_norm.t()
            # dist /= torch.mm(x_norm, y_norm)

            # print(dist.size())
            # print(dist.size(), (x_norm / y_norm.t()).size())
            # print(dist)

            # print(dist)
            # print(dist.size())
            # X = np.matmul(forward, forward.T)
            # m = np.diag(X)
            # m = np.matmul(np.expand_dims(m, -1), np.ones((1, len(forward))))
            # d = m + m.T - 2 * X

            # dist = torch.mul(dist, from_numpy(W).float().to(self.device))

            # print(dist)
            # print(tot_p)
            # a = torch.tensor(torch.tensor(np.mean(d, keepdims=True), requires_grad=True)).float()
            # t = torch.Tensor(a)
            # t.backward()
            mx = torch.mean(dist, 1)
            # print(mx)
            mx = 1 - mx
            # print(mx)
            # print(mx)
            # mx = torch.log1p(mx)
            # print(mx)
            mn = -torch.log(torch.mean(mx))
            # mn.backward()

            # mn.backward()
            # print(sub_task, mn)
            # w = 1 ** -sub_task
            # print(sub_task, w)
            tot_loss += mn #* (1 / (sub_task+1))

        tot_loss.backward()
        # mn = torch.log(tot_loss)
        # # print(mn)
        # # mn.backward()
        #     for n, p in self.model.named_parameters():
        #         if p.requires_grad:
        #             # mx = torch.max(dist, 1)[0]
        #             g = grad(mn, p, allow_unused=True, retain_graph=True)[0]
        #             # print(n, g)
        #             if g is None:
        #                 continue
        #             p.grad.add_(g)
                    # print(n, loss)
        # print(tot_loss)
        # print(t.grad_fn)
        # getBack(t.grad_fn)
        return 0


