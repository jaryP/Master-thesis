import quadprog
from copy import deepcopy
from networks.net_utils import AbstractNetwork
from utils.datasetsUtils.dataset import GeneralDatasetLoader
import torch.nn.functional as F
import random
from torch import stack, cat, mm, Tensor, clamp, zeros_like, from_numpy, unsqueeze, matmul, mul, abs, nn
import torch
import numpy as np
from scipy.stats import multivariate_normal


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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
        self.importance = config.CL_PAR.get('penalty_importance', 1)

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

            if self.is_incremental:
                 self.model.task = self.dataset.task_mask(task_idx)
            else:
                 self.model.task = task_idx

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
        self.batch_size = config.BATCH_SIZE

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self._current_task = 0

        self.margin = config.CL_PAR.get('margin', 0.5)
        self.sample_size = config.CL_PAR.get('sample_size', 50)

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
                current_gradients[n] = deepcopy(p.grad.data.view(-1).cpu())

        self._save_gradients(current_task=kwargs['current_task'])

        done = self._constraints_gradients(current_task=kwargs['current_task'], current_gradients=current_gradients)

        if not done:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.grad.copy_(current_gradients[n].view(p.grad.data.size()).cpu())

        return self, 0

    def _qp(self, past_tasks_gradient, current_gradient):
        t = past_tasks_gradient.shape[0]
        P = np.dot(past_tasks_gradient, past_tasks_gradient.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
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
            a = mm(cg.unsqueeze(0), tg)

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

            self.model.task = sub_task
            self.model.zero_grad()

            loss = 0
            for images, label in self.dataset.getIterator(self.sample_size, task=sub_task):

                input = images.to(self.device)
                label = label.to(self.device)

                for i in chunks(range(self.sample_size), self.batch_size):
                    inp = input[i]
                    l = label[i]
                    output = self.model(inp)
                    loss += F.nll_loss(F.log_softmax(output, dim=1), l)

                break

            loss.backward()

            gradients = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    gradients[n] = p.grad.data.view(-1).cpu()

            self.tasks_gradients[sub_task] = deepcopy(gradients)


class GEM_MEM(object):
    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset
        self.is_incremental = config.IS_INCREMENTAL
        self.batch_size = config.BATCH_SIZE

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self._current_task = 0

        self.margin = config.CL_PAR.get('margin', 0.5)
        self.memorized_task_size = config.CL_PAR.get('memorized_task_size', 300)
        self.sample_size = config.CL_PAR.get('sample_size', self.memorized_task_size)

        self.current_gradients = {}
        self.tasks_gradients = {}

        self.tasks = []
        self._precision_matrices = None
        self.memory = {}
        self.loss_f = nn.CrossEntropyLoss()

    def __call__(self, *args, **kwargs):

        if kwargs['current_task'] == 0:
            return self, 0

        current_gradients = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                current_gradients[n] = deepcopy(p.grad.data.view(-1).cpu())

        self._save_gradients(current_task=kwargs['current_task'])

        done = self._constraints_gradients(current_task=kwargs['current_task'], current_gradients=current_gradients)

        if not done:
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    p.grad.copy_(current_gradients[n].view(p.grad.data.size()).cpu())

        return self, 0

    def _qp(self, past_tasks_gradient, current_gradient):
        t = past_tasks_gradient.shape[0]
        P = np.dot(past_tasks_gradient, past_tasks_gradient.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
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
            a = mm(cg.unsqueeze(0), tg)

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

            self.model.task = sub_task
            self.model.zero_grad()

            if sub_task not in self.memory:
                images, label = next(self.dataset.getIterator(self.sample_size, task=sub_task))
                self.memory[sub_task] = {'images': images, 'labels': label}

            input = self.memory[sub_task]['images'].to(self.device)
            label = self.memory[sub_task]['labels'].to(self.device)

            idx = range(len(input))
            if self.sample_size != self.memorized_task_size:
                idx = random.choices(idx, k=self.sample_size)

            for i in chunks(idx, self.batch_size):
                inp = input[i]
                l = label[i]
                self.loss_f(self.model(inp), l).backward()

            gradients = {}
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    gradients[n] = p.grad.data.view(-1).cpu()

            self.tasks_gradients[sub_task] = deepcopy(gradients)


class embedding(object):
    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config):

        self.model = model
        self.dataset = dataset

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.batch_size = config.BATCH_SIZE
        self.first_batch = True
        self.incremental = config.IS_INCREMENTAL
        self.handle = None

        self.sample_size = config.CL_PAR.get('sample_size', 25)
        self.memorized_task_size = config.CL_PAR.get('memorized_task_size', 300)
        self.importance = config.CL_PAR.get('penalty_importance', 1)
        self.distance = config.CL_PAR.get('distance', 'euclidean')
        self.supervised = config.CL_PAR.get('supervised', True)
        self.normalize = config.CL_PAR.get('normalize', True)

        # Can be distance, usage, image_similarity or none
        self.weights_type = config.CL_PAR.get('weights_type', None)

        if self.weights_type == 'image_similarity':
            img_size = dataset[0][0].size()
            if self.is_conv:
                pass
            else:
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(28 * 28, 300),
                    torch.nn.ReLU(),
                    torch.nn.Linear(300, 200),
                    torch.nn.ReLU(),
                    torch.nn.Linear(200, 100),
                ).to(config.DEVICE)


        self._current_task = 0
        self.sample_size_task = 0
        self.batch_count = 0

        self.tasks = set()

        self.embeddings = {}
        self.embeddings_images = {}
        self.w = {}

    def __call__(self, *args, **kwargs):

        if self.importance == 0:
            return self, 0

        penalty = 0
        current_task = kwargs['current_task']

        if current_task > 0:
            
            kwargs['optimizer'].step()
            current_batch = kwargs['batch']

            self.batch_count += 1
            self.embedding_save(current_task)

            penalty = self.embedding_drive(current_batch)


        return self, penalty

    def embedding_save(self, current_task):

        if current_task - 1 not in self.tasks:
            self.first_batch = True
            self.tasks.add(current_task - 1)

            self.batch_count = 0

            self.model.eval()
            self.dataset.train_phase()

            it = self.dataset.getIterator(self.memorized_task_size, task=current_task - 1)

            images, _ = next(it)

            input = images.to(self.device)
            embs = None

            with torch.no_grad():

                for i in input:
                    output = self.model.embedding(i.unsqueeze(0))

                    if self.normalize:
                        output = F.normalize(output, p=2, dim=1)

                    embeddings = output.cpu()

                    if embs is None:
                        embs = embeddings
                    else:
                        embs = torch.cat((embs, embeddings), 0)

                self.embeddings[current_task - 1] = embs.cpu()
                self.embeddings_images[current_task - 1] = images.cpu()

                self.w[current_task - 1] = [1] * self.memorized_task_size

            self.model.train()
            self.dataset.train_phase()

    def embedding_drive(self, current_batch):
        self.model.eval()
        
        for t in self.embeddings:

            to_back = []

            w = self.w[t]

            idx = range(len(w))

            if self.weights_type == 'usage':
                w = [1 / wc for wc in w]

            ss = self.sample_size
          
            idx = random.choices(idx, k=ss, weights=w)

            for i in chunks(idx, self.batch_size):
                img = self.embeddings_images[t][i].to(self.device)
                embeddings = self.embeddings[t][i]

                new_embeddings = self.model.embedding(img)

                if self.normalize:
                    new_embeddings = F.normalize(new_embeddings, p=2, dim=1)

                new_embeddings = new_embeddings.cpu()

                if self.distance == 'euclidean':
                    dist = (embeddings - new_embeddings).norm(p=None, dim=1)
                elif self.distance == 'cosine':
                    cosine = torch.nn.functional.cosine_similarity(embeddings, new_embeddings)
                    dist = 1 - cosine

                to_back.append(dist)

                if self.weights_type is not None:
                    if self.weights_type == 'distance':
                        dist = dist.detach().cpu().numpy()
                        for j, k in enumerate(i):
                            self.w[t][j] = dist[j]

                    elif self.weights_type == 'usage':
                        for j, k in enumerate(i):
                            self.w[t][k] += 1

                    elif self.weights_type == 'image_similarity':
                        current_images = self.encoder(current_batch[0])
                        old_images = self.encoder(img)

                        with torch.no_grad():
                            current_images = current_images / current_images.norm(dim=1)[:, None]
                            old_images = old_images / old_images.norm(dim=1)[:, None]
                            dist = torch.mm(current_images, old_images.transpose(0, 1))
                            dist = (1 - dist.mean(dim=1)).cpu().numpy()

                        for j, k in enumerate(i):
                            self.w[t][k] = dist[j]

            torch.mul(torch.cat(to_back, 0).mean(), self.importance).backward()

        self.model.train()
        return 0
