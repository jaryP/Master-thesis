from sklearn.metrics import f1_score, accuracy_score


def calculate_all_metrics(y_true, y_predicted):
    d = dict()
    s = set(y_true)
    is_binary = True if len(s) == 2 else True

    d['f1'] = f1(y_true, y_predicted, is_binary)
    d['accuracy'] = accuracy_score(y_true, y_predicted)

    return d


def f1(y_true, y_predicted, is_binary=True):
    if is_binary:
        return f1_score(y_true, y_predicted)
    else:
        return f1_score(y_true, y_predicted, average='micro')


def accuracy(y_true, y_predicted):
    return accuracy_score(y_true, y_predicted)
