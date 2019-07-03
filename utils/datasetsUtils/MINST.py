import gzip
import codecs
from torch.utils.data import BatchSampler, RandomSampler
import utils.datasetsUtils.dataset
from utils.datasetsUtils.taskManager import AbstractTaskDecorator, NoTask, DuplicatetNoTask
import urllib.request
import shutil
from os.path import join
import numpy as np
import torch
import random
from scipy.ndimage import rotate


class MINST(utils.datasetsUtils.dataset.GeneralDatasetLoader):

    def __init__(self, folder: str, task_manager: AbstractTaskDecorator, train_split: float = 0.9,
                 transform=None, target_transform=None, download=False,
                 force_download=False):

        super().__init__(folder)

        self.url = \
            [
                'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
            ]

        self.filename = "cifar-10-python.tar.gz"
        self.unzipped_folder = ''

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
            task = self.task

        t = self.task2idx[task][self._phase]
        img = self.X[t['x'][index]]
        target = t['y'][index]

        # img = Image.fromarray(img.numpy(), mode='L')
        img = torch.from_numpy(np.array(img, copy=False, dtype=np.float)).float()

        # if len(list(img.size())) < 3:
        #     img = img.unsqueeze(0)

        if self.transform is not None:
            img = self.transform(img)

        if isinstance(target, int):
            target = [target]

        target = np.asarray(target, dtype=np.int64)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = torch.from_numpy(target)

        return img, target

    def __len__(self):
        return len(self.task2idx[self._current_task][self._phase]['x'])

    def getIterator(self, batch_size, task=None):

        if task is None:
            task = self.task

        class CustomIterator:
            def __init__(self, outer_dataset):

                self.minst = outer_dataset
                self.sampler = BatchSampler(RandomSampler(range(len(outer_dataset.task2idx
                                                                    [task]
                                                                    [outer_dataset.phase]
                                                                    ['x']))),
                                            batch_size, False)

            def __iter__(self):
                for batch_idx in self.sampler:
                    x = []
                    y = []
                    for i in batch_idx:
                        xc, yc = self.minst[(i, task)]
                        x.append(xc)
                        y.append(yc)

                    x = torch.stack(x)
                    y = torch.stack(y)
                    y = y.squeeze(1)

                    yield x, y

        return iter(CustomIterator(self))

    def _read_label_file(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            length = self._get_int(data[4:8])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
            parsed = parsed.reshape((length, -1))
            parsed = np.squeeze(parsed)

        return parsed.tolist()

    def _get_int(self, b):
        return int(codecs.encode(b, 'hex'), 16)

    def _load_image(self, path):
        with open(path, 'rb') as f:
            data = f.read()
            length = self._get_int(data[4:8])
            num_rows = self._get_int(data[8:12])
            num_cols = self._get_int(data[12:16])
            parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
            parsed = parsed.reshape((length, num_rows, num_cols))

        return parsed

    def load_dataset(self, ):

        if self.download:
            self.download_dataset()

        train_list = [('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')]
        test_list = [('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')]

        downloaded_list = train_list[::]
        downloaded_list.extend(test_list)
        X = []
        Y = []

        for (images, labels) in downloaded_list:
            
            images_path = join(self.folder, self.unzipped_folder, images)
            labels_path = join(self.folder, self.unzipped_folder, labels)

            a = self._load_image(images_path)
            l = self._read_label_file(labels_path)

            X.append(a)
            Y.extend(l)

        X = np.vstack(X)
        X = np.reshape(X, (-1, X.shape[1]*X.shape[2]))

        # X = X.transpose((0, 2, 3, 1))
        labels = set([i for i in Y])

        self.class_to_idx = {_class: i for i, _class in enumerate(labels)}
        self.idx_to_class = {_class: i for i, _class in enumerate(labels)}

        self.X, self.Y = X, Y
        if self.task_manager is not None:
            task_map = self.task_manager.process_idx(ground_truth_labels=Y)
        else:
            task_map = NoTask().process_idx(ground_truth_labels=Y)

        self.task2idx = self.train_test_split(task_map)
        self._n_tasks = len(self.task2idx)

        for t, d in self.task2idx.items():
            print('task {} with train {} and test {} images (label: {})'.format(t, len(d['train']['x']),
                                                                                 len(d['test']['x']),
                                                                                 0))

    def train_test_split(self, task_map):

        return_dict = dict()

        for task, d in task_map.items():
            x = d['x']
            y = d['y']

            idx = list(range(len(x)))
            np.random.shuffle(idx)
            cut = int(len(idx) * self.train_split)

            return_dict[task] = {'train': {'x': x[:cut], 'y': y[:cut]},
                                 'test': {'x': x[cut:], 'y': y[cut:]}}

        return return_dict

    def download_dataset(self):

        downloaded = self.already_downloaded()
        if not self.force_download and downloaded:
            return

        for url in self.url:
            filename = url.rpartition('/')[2]
            gzip_path = join(self.download_path, filename)

            with urllib.request.urlopen(url) as response, open(gzip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

        for url in self.url:
            filename = url.rpartition('/')[2]
            gzip_path = join(self.download_path, filename)
            with open(join(self.folder, filename).replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(gzip_path) as zip_f:

                out_f.write(zip_f.read())


class PermutedMINST(MINST):
    def __init__(self, folder: str, train_split: float = 0.9,
                 transform=None, target_transform=None, download=False,
                 force_download=False, n_permutation=2):

        super().__init__(folder, DuplicatetNoTask(n_permutation), train_split, transform, target_transform, download, force_download)
        self.permuted_index = []
        self.n_permutation = n_permutation
        self._n_task = n_permutation

    def __getitem__(self, index):

        if isinstance(index, list) or isinstance(index, tuple):
            index, task = index
        else:
            task = self.task

        t = self.task2idx[task][self._phase]
        img = self.X[t['x'][index]]
        img = img[self.permuted_index[task]]
        target = t['y'][index]

        img = torch.from_numpy(img).float()

        if self.transform is not None:
            img = self.transform(img)

        if isinstance(target, int):
            target = [target]

        target = np.asarray(target, dtype=np.int64)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = torch.from_numpy(target)

        return img, target

    def load_dataset(self):
        super().load_dataset()

        self.X = [x/255 for x in self.X]
        shape = self.X[0].shape[0]

        idx = list(range(shape))

        for i in range(self.n_permutation):
            self.permuted_index.append([idx.copy()])
            random.shuffle(idx)


class RotatedMINST(MINST):
    def __init__(self, folder: str, train_split: float = 0.9, task_manager=None,
                 transform=None, target_transform=None, download=False,
                 force_download=False, n_rotations=2):

        # if task_manager is None:
        #     task_manager = DuplicatetNoTask(n_rotations)

        super().__init__(folder, DuplicatetNoTask(n_rotations), train_split, transform, target_transform, download, force_download)
        self.angle = []
        self.n_rotations = n_rotations
        self._n_task = n_rotations

    def __getitem__(self, index):

        if isinstance(index, list) or isinstance(index, tuple):
            index, task = index
        else:
            task = self.task

        t = self.task2idx[task][self._phase]
        img = self.X[t['x'][index]]
        angle = self.angle[task]
        target = t['y'][index]

        img = rotate(img.reshape(28, 28), angle, reshape=False).reshape(-1)

        img = torch.from_numpy(img).float()

        if self.transform is not None:
            img = self.transform(img)

        if isinstance(target, int):
            target = [target]

        target = np.asarray(target, dtype=np.int64)

        if self.target_transform is not None:
            target = self.target_transform(target)
        else:
            target = torch.from_numpy(target)

        return img, target

    def load_dataset(self):
        super().load_dataset()

        self.X = [x/255 for x in self.X]
        self.angle.append(0)

        for i in range(1, self.n_rotations):
            self.angle.append(random.uniform(-180, 180))
