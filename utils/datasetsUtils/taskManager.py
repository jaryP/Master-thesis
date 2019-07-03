from abc import ABC, abstractmethod


class AbstractTaskDecorator(ABC):
    @staticmethod
    @abstractmethod
    def process_idx(*args, **kwargs):
        pass


class NoTask(AbstractTaskDecorator):

    @staticmethod
    def process_idx(*args, **kwargs):
        ground_truth_labels = kwargs['ground_truth_labels']
        x = list(range(len(ground_truth_labels)))

        return {0: {'x': x, 'y': ground_truth_labels}}


class SingleTargetClassificationTask(AbstractTaskDecorator):

    @staticmethod
    def process_idx(*args, **kwargs):
        ground_truth_labels = kwargs['ground_truth_labels']

        labels_set = set(ground_truth_labels)
        idx_per_task = dict()

        indexes = list(range(len(ground_truth_labels)))

        for gt in labels_set:
            labels = [1 if l == gt else 0 for l in ground_truth_labels]
            idx_per_task[gt] = {'y': labels,
                                'x': indexes}

        return idx_per_task


class DuplicatetNoTask(AbstractTaskDecorator):
    def __init__(self, n):
        self.n = n

    def process_idx(self, *args, **kwargs):
        ground_truth_labels = kwargs['ground_truth_labels']
        d = {}
        for i in range(self.n):
           d[i] = {'x': list(range(len(ground_truth_labels))), 'y': ground_truth_labels}
        return d


class IncrementalTaskClassification(AbstractTaskDecorator):

    def __init__(self, incremental_size=2):
        self.incremental_size = incremental_size

    def process_idx(self, *args, **kwargs):
        ground_truth_labels = kwargs['ground_truth_labels']

        ground_truth_labels_set = sorted(list(set(ground_truth_labels)))
        groups = [ground_truth_labels_set[i:i + self.incremental_size]
                  for i in range(0, len(ground_truth_labels_set), self.incremental_size)]

        if len(groups[-1]) == 1:
            groups[-2] = groups[-2] + groups[-1]
            groups = groups[:-1]

        indexes = list(range(len(ground_truth_labels)))

        idx_per_task = dict()

        for ti, g in enumerate(groups):
            gl = []
            gi = []
            for i in indexes:
                gtl = ground_truth_labels[i]
                if gtl in g:
                    gl.append(gtl)
                    gi.append(i)

            idx_per_task[tuple(g)] = {'y': gl,
                                'x': gi}

        return idx_per_task
