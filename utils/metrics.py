from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import numpy as np
import json


class MetricsHolder:
    def __init__(self, n_task):

        self._r = np.empty((n_task, n_task))
        self._r.fill(-1)

        self._metrics = defaultdict(list)
        self._tasks = defaultdict(lambda: defaultdict(list))

    def add_evaluation(self, evaluated_task, current_task, y_true, y_pred):
        is_binary = True if len(set(y_true)) == 2 else False
        if evaluated_task is None:
            evaluated_task = current_task

        acc = accuracy(y_true, y_pred)

        self._r[current_task, evaluated_task] = acc

        if evaluated_task <= current_task:
            self._tasks[evaluated_task]['f1'].append(f1(y_true, y_pred, is_binary))
            self._tasks[evaluated_task]['accuracy'].append(acc)

    @property
    def metrics(self):
        return self._metrics

    @metrics.getter
    def metrics(self):
        self._metrics['fwt'] = fwt(self._r)
        self._metrics['bwt'], self._metrics['remembering'], self._metrics['pbwt'] = bwt(self._r)
        self._metrics['accuracy'] = total_accuracy(self._r)

        return {'metrics':  json.loads(json.dumps(self._metrics)), 'tasks':  json.loads(json.dumps(self._tasks))}


def bwt(r):

    n = r.shape[0]
    v = 0

    if n == 1:
        return v, 1 - abs(min(v, 0)), max(v, 0)

    for i in range(1, n):
        for j in range(i):
            v += r[i][j] - r[j][j]

    v = v / ((n*(n-1))/2)

    return v, 1 - abs(min(v, 0)), max(v, 0)


def fwt(r):
    n = r.shape[0]
    v = 0

    if n == 1:
        return v

    for i in range(n):
        for j in range(i):
            v += r[i][j]

    v = v / ((n * (n - 1)) / 2)
    return v


def total_accuracy(r):
    n = r.shape[0]
    v = 0

    for i in range(n):
        for j in range(i, n):
            v += r[i][j]

    v = v / ((n * (n + 1)) / 2)
    return v


def accuracy(y_true, y_predicted):
    return accuracy_score(y_true, y_predicted)

#
# def calculate_all_metrics(y_true, y_predicted):
#     d = dict()
#     s = set(y_true)
#     is_binary = True if len(s) == 2 else True
#
#     d['f1'] = f1(y_true, y_predicted, is_binary)
#     d['accuracy'] = accuracy_score(y_true, y_predicted)
#
#     return d


def f1(y_true, y_predicted, is_binary=True):
    if is_binary:
        return f1_score(y_true, y_predicted)
    else:
        return f1_score(y_true, y_predicted, average='micro')
