from torch.utils.data import BatchSampler, RandomSampler
import utils.datasetsUtils.dataset
from utils.datasetsUtils.taskManager import AbstractTaskDecorator, NoTask
import tarfile
import urllib.request
import shutil
from os.path import join
import pickle
import numpy as np
from PIL import Image
import torch


class Cifar10(utils.datasetsUtils.dataset.GeneralDatasetLoader):

    def __init__(self, folder: str, task_manager: AbstractTaskDecorator, train_split: float = 0.9,
                 transform=None, target_transform=None, download=False,
                 force_download=False):

        super().__init__(folder)

        self.url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.filename = "cifar-10-python.tar.gz"
        self.unzipped_folder = 'cifar-10-batches-py'

        self.transform = transform
        self.target_transform = target_transform

        self._phase = 'train'
        self._current_task = 0

        self.download = download
        self.force_download = force_download

        self.train_split = train_split

        self.task_manager = task_manager

        self.X, self.Y, self.class2idx, self.idx2class = None, None, None, None
        self.task2idx = None
        self.task_map = None
        self._n_tasks = None
        self.class_to_idx = None
        self.idx_to_class = None
        self._n_classes = 0

    def __getitem__(self, index):

        if isinstance(index, list) or isinstance(index, tuple):
            index, task = index
        else:
            task = self._current_task

        t = self.task2idx[task][self._phase]
        img = self.X[t['x'][index]]
        target = t['y'][index]

        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if isinstance(target, int):
            target = [target]

        # y = np.zeros(self._n_classes)
        # y[target] = 1
        target = np.asarray(target, dtype=np.int64)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = torch.from_numpy(target)

        # img = torch.tensor(img)
        # if len(list(img.size)) < 4:
        #     img = img.unsqueeze(0)

        return img, target

    def __len__(self):
        task = list(self.task2idx.keys())[self._current_task]
        return len(self.task2idx[task][self._phase]['x'])

    def getIterator(self, batch_size, task=None):

        class CustomIterator:
            def __init__(self, outer_dataset):

                self.cifar = outer_dataset
                if task is not None:
                    self.t = list(outer_dataset.task2idx.keys())[task]
                    # self.t = task
                else:
                    self.t = list(outer_dataset.task2idx.keys())[outer_dataset.task]

                    # self.t = outer_dataset.task

                self.sampler = BatchSampler(RandomSampler(range(len(outer_dataset.task2idx
                                                                    [self.t]
                                                                    [outer_dataset.phase]
                                                                    ['x']))),
                                            batch_size, False)

            def __iter__(self):
                for batch_idx in self.sampler:
                    x = []
                    y = []
                    for i in batch_idx:
                        xc, yc = self.cifar[(i, self.t)]
                        x.append(xc)
                        y.append(yc)

                    x = torch.squeeze(torch.stack(x), 1)
                    y = torch.squeeze(torch.stack(y), 1)

                    yield x, y

        return iter(CustomIterator(self))

    def load_dataset(self):

        if self.download:
            self.download_dataset()

        path = join(self.folder, self.unzipped_folder, 'batches.meta')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')

        X = []
        Y = []

        self.class_to_idx = {_class: i for i, _class in enumerate(data['label_names'])}
        self.idx_to_class = {i: _class for i, _class in enumerate(data['label_names'])}
        self._n_classes = len(self.idx_to_class)

        train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_list = ['test_batch']

        downloaded_list = train_list[::]
        downloaded_list.extend(test_list)

        for file_name in downloaded_list:
            file_path = join(self.folder, self.unzipped_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                X.append(entry['data'])

                if 'labels' in entry:
                    Y.extend(entry['labels'])
                else:
                    Y.extend(entry['fine_labels'])

        X = np.vstack(X).reshape(-1, 3, 32, 32)
        X = X.transpose((0, 2, 3, 1))

        self.X, self.Y = X, Y

        if self.task_manager is not None:
            task_map = self.task_manager.process_idx(ground_truth_labels=Y)
        else:
            task_map = NoTask().process_idx(ground_truth_labels=Y)

        self.task2idx = self.train_test_split(task_map)

        self._n_tasks = len(self.task2idx)

        # for t, d in self.task2idx.items():
        #     print('task #{} with train {} and test {} images (label: {})'.format(t, len(d['train']['x']), len(d['test']['x']),
        #                                                                          0))

    def train_test_split(self, task_map):

        return_dict = dict()

        for task, d in task_map.items():
            x = d['x']
            y = d['y']

            idx = list(range(len(x)))
            np.random.shuffle(idx)
            cut = int(len(idx)*self.train_split)

            return_dict[task] = {'train': {'x': x[:cut], 'y': y[:cut]},
                                 'test': {'x': x[cut:], 'y': y[cut:]}}

        return return_dict

    def download_dataset(self):

        if not self.force_download:
            if self.already_downloaded():
                return

        with urllib.request.urlopen(self.url) as response, open(join(self.download_path, self.filename),
                                                                'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        with tarfile.open(join(self.download_path, self.filename), "r:gz") as tar:
            tar.extractall(path=self.folder)


class Cifar100(Cifar10):

    def __init__(self, folder: str, task_manager: AbstractTaskDecorator, train_split: float = 0.9,
                 transform=None, target_transform=None, download=False,
                 force_download=False, superclasses=False):

        super().__init__(folder=folder, task_manager=task_manager, train_split=train_split, transform=transform,
                         target_transform=target_transform, download=download, force_download=force_download)

        self.url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        self.filename = "cifar-100-python.tar.gz"
        self.unzipped_folder = 'cifar-100-python'

        if not superclasses:
            self.classes = 'fine_label_names'
        else:
            self.classes = 'coarse_label_names'

    def load_dataset(self):

        if self.download:
            self.download_dataset()

        path = join(self.folder, self.unzipped_folder, 'meta')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')

        X = []
        Y = []

        self.class_to_idx = {_class: i for i, _class in enumerate(data[self.classes])}
        self.idx_to_class = {i: _class for i, _class in enumerate(data[self.classes])}

        if self.classes == 'fine_label_names':
            class_to_extract = 'fine_labels'
        else:
            class_to_extract = 'coarse_labels'

        train_list = ['train']
        test_list = ['test']

        downloaded_list = train_list[::]
        downloaded_list.extend(test_list)

        for file_name in downloaded_list:
            file_path = join(self.folder, self.unzipped_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                X.append(entry['data'])
                if 'labels' in entry:
                    Y.extend(entry['labels'])
                else:
                    Y.extend(entry[class_to_extract])

        X = np.vstack(X).reshape(-1, 3, 32, 32)
        X = X.transpose((0, 2, 3, 1))

        self.X, self.Y = X, Y

        if self.task_manager is not None:
            task_map = self.task_manager.process_idx(ground_truth_labels=Y)
        else:
            task_map = NoTask().process_idx(ground_truth_labels=Y)

        self.task2idx = self.train_test_split(task_map)

        self._n_tasks = len(self.task2idx)

        # for t, d in self.task2idx.items():
        #     print('task #{} with train {} and test {} images (label: {})'.format(t, len(d['train']), len(d['test']),
        #                                                                          self.idx_to_class[t]))


