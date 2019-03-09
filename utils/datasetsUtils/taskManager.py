from collections import defaultdict
from abc import ABC, abstractmethod


class AbstractTaskDecorator(ABC):
    @staticmethod
    @abstractmethod
    def process_idx(*args, **kwargs):
        pass


class NoTask(AbstractTaskDecorator):
    @staticmethod
    def process_idx(x, groud_truth_labels):
        return {0: {'x': x, 'y': groud_truth_labels}}


class SingleTargetClassificationTask(AbstractTaskDecorator):
    @staticmethod
    def process_idx(x, groud_truth_labels):

        labels_set = set(groud_truth_labels)
        idx_per_task = dict()

        indexes = list(range(len(groud_truth_labels)))

        for gt in labels_set:
            labels = [1 if l == gt else 0 for l in groud_truth_labels]
            idx_per_task[gt] = {'y': labels,
                                'x': indexes}

        return idx_per_task

