import collections

from tqdm import tqdm
import configs.configClasses as configClasses
from utils.datasetsUtils.dataset import GeneralDatasetLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import optim, abs, save, load, nn
import torch
from utils.metrics import MetricsHolder, accuracy, f1
from networks.net_utils import AbstractNetwork
from os.path import join, exists, isdir
from os import makedirs
import warnings


class Trainer:
    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config: configClasses.DefaultConfig,
                 save_modality=1, verbose=True, pretrained_model=None):

        self.config = config
        self.dataset = dataset
        self.model = model
        self.metrics_calculator = MetricsHolder(self.dataset.tasks_number)
        self.verbose = verbose
        self.pretrained_model = pretrained_model

        self.epochs_n = self.config.EPOCHS
        self.save_modality = save_modality
        self.device = config.DEVICE
        self.is_incremental = config.IS_INCREMENTAL

        self.next_task_lr = config.NEXT_TASK_LR
        self.next_task_epochs = config.NEXT_TASK_EPOCHS

        self.save_path = join(config.SAVE_PATH, config.MODEL_NAME)
        if not exists(config.SAVE_PATH):
            makedirs(config.SAVE_PATH)

        if config.DEVICE != 'cpu':
            self.model.to(self.device)

        self.cont_learn_tec = None
        if config.USE_CL:
            if config.CL_TEC is None:
                warnings.warn("Ewc type is set to None  ")
            else:
                self.cont_learn_tec = config.CL_TEC(self.model, self.dataset, config)

        self.results = dict()

        if config.LOSS == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError('Not known loss function, allowed ones are: cross_entropy')

        if config.OPTIMIZER == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.LR)
        elif config.OPTIMIZER == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR)
        elif config.OPTIMIZER == 'ASGD':
            self.optimizer = optim.ASGD(self.model.parameters(), lr=self.config.LR)
        else:
            raise ValueError('Not known optimizer, allowed ones are: SGD')

    def single_task(self, task=0):
        losses = []

        self.dataset.task = task

        for e in range(self.config.EPOCHS):
            loss = self.epoch(e)
            losses.append(loss)
            self.evaluate(task)

        if self.save_modality >= 1:
            state = {'metrics': self.metrics_calculator.metrics, 'losses': losses,
                     'model': self.model.state_dict()}
            save(state, self.save_path + "_last")

        return {'losses': losses, 'metrics': self.metrics_calculator.metrics}

    def all_tasks(self, limit=-1):

        has_next_task = True
        losses_per_task = dict()

        while has_next_task:

            if self.config.OPTIMIZER == 'Adam' and (self.is_incremental or self.cont_learn_tec is not None):
                self.optimizer.stat = collections.defaultdict(dict)

            current_task = self.dataset.task

            if current_task == 1:

                if self.next_task_epochs is not None:
                    self.epochs_n = self.next_task_epochs
                if self.next_task_lr is not None:
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.next_task_lr

            if 0 < limit <= current_task:
                break

            losses = []

            # if (current_task > 0) and self.cont_learn_tec is not None:
            #     self.cont_learn_tec(current_task=current_task)

            self.dataset.task = current_task

            if self.is_incremental:
                self.model.task = self.dataset.task_mask(current_task)
                self.model.used_tasks.update(set(self.dataset.task_mask(current_task)))
            else:
                self.model.task = current_task
                self.model.used_tasks.add(current_task)

            # if self.is_incremental:
            #     pass
            #     # self.model.task = self.dataset.task_mask(current_task)
            # else:
            #     pass

            for e in range(self.epochs_n):
                loss = self.epoch(e)
                losses.append(loss)

                self.evaluate(current_task)

                if current_task > 0 and self.dataset.tasks_number > 1:
                    for sub_task in range(current_task):
                        self.evaluate(current_task, sub_task)

                # for t, v in self.metrics_calculator.metrics['tasks'].items():
                #     print(t, v['accuracy'])

                self.dataset.task = current_task

            if self.dataset.tasks_number > 1:
                for sub_task in range(current_task+1, self.dataset.tasks_number):
                    self.evaluate(current_task, sub_task)

            losses_per_task[current_task] = losses
            has_next_task = self.dataset.next_task(round_robin=False)

            if self.save_modality >= 2:
                state_dict = {}
                for k, v in self.model.state_dict().items():
                    state_dict[k] = v.cpu()

                m = self.metrics_calculator.metrics
                state = {'metrics': m['metrics'], 'tasks': m['tasks'], 'losses': losses_per_task,
                         'model': state_dict}

                save(state, self.save_path+"_"+str(current_task))

        if self.save_modality >= 1:
            state_dict = {}
            for k, v in self.model.state_dict().items():
                state_dict[k] = v.cpu()

            m = self.metrics_calculator.metrics
            state = {'metrics': m['metrics'], 'tasks': m['tasks'], 'losses': losses_per_task,
                     'model': self.model.state_dict()}

            save(state, self.save_path + "_last")

        metrics = self.metrics_calculator.metrics
        metrics['losses'] = losses_per_task

        return metrics

    def epoch(self, n):
        self.model.train()
        self.dataset.train_phase()
        current_task = self.dataset.task

        i = 0
        epoch_loss_full = 0

        it = tqdm(self.dataset.getIterator(self.config.BATCH_SIZE), total=len(self.dataset)//self.config.BATCH_SIZE,
                  disable=not self.verbose)

        if self.cont_learn_tec is not None:
            it.set_description("Training task ({}) {}, epoch {}".format(self.cont_learn_tec.__class__.__name__,
                                                                        self.dataset.task, n+1))
        else:
            it.set_description("Training task {}, epoch {}".format(self.dataset.task, n+1))

        for x, y in it:

            if self.is_incremental:
                self.model.task = self.dataset.task_mask(current_task)
            else:
                self.model.task = current_task

            x, y = x.to(self.config.DEVICE), \
                            y.to(self.config.DEVICE)

            self.optimizer.zero_grad()
            output = self.model(x)

            loss = self.loss_function(output, y)

            if self.config.L1_REG > 0:
                l1_loss = 0.0
                for name, param in self.model.named_parameters():
                    l1_loss += torch.sum(abs(param))
                loss = loss + self.config.L1_REG * l1_loss

            loss.backward(retain_graph=True)
            # print('trainer 1', self.model.classification_layer.bias.grad)

            if self.cont_learn_tec is not None:
                if self.dataset.task > 0:
                    a = 0

                self.cont_learn_tec, penalty = self.cont_learn_tec(current_task=self.dataset.task, batch=(x, y))

                if penalty != 0:

                    loss = loss + penalty
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)

            # print('trainer 2', self.model.classification_layer.bias.grad)

            clip_grad_norm_(self.model.parameters(), 20)
            self.optimizer.step()
            epoch_loss_full += loss.detach().item()
            i += 1

            if self.cont_learn_tec is None:
                it.set_postfix({'loss': epoch_loss_full/i,
                                'batch#': i})
            else:
                it.set_postfix({'loss': epoch_loss_full / i, 'batch#': i})

        return epoch_loss_full/i

    def evaluate(self, current_task, evaluated_task=None):
        self.dataset.test_phase()
        self.model.eval()

        evaluated_task = evaluated_task if evaluated_task is not None else current_task

        if self.is_incremental:
            self.model.task = self.dataset.task_mask(evaluated_task)
        else:
            self.model.task = evaluated_task

        if evaluated_task is None:
            evaluated_task = current_task

        i = 0

        it = tqdm(self.dataset.getIterator(self.config.BATCH_SIZE, task=evaluated_task), disable=not self.verbose)
        it.set_description("Current task: {}, evaluated task {}".format(current_task, evaluated_task))

        y_true = []
        y_pred = []

        for x, y in it:
            x, y = x.to(self.config.DEVICE), \
                   y.to(self.config.DEVICE)

            with torch.no_grad():
                y_pred_np = self.model.eval_forward(x).cpu().numpy()
                y_true_np = y.cpu().numpy()

            y_true.extend(list(y_true_np))
            y_pred.extend(list(y_pred_np))
            i += 1

            it.set_postfix({'batch#': i})

        self.metrics_calculator.add_evaluation(evaluated_task=evaluated_task, current_task=current_task,
                                               y_true=y_true, y_pred=y_pred)

    def evaluate_on_dataset(self, dataset):
        dataset.test_phase()
        self.model.eval()

        has_next_task = True

        y_true = []
        y_pred = []

        while has_next_task:
            # evaluated_task = evaluated_task if evaluated_task is not None else current_task
            current_task = dataset.task

            # if self.is_incremental:
            #     self.model.task = self.dataset.task_mask(evaluated_task)
            # else:
            self.model.task = current_task

            # if evaluated_task is None:
            #     evaluated_task = current_task

            i = 0

            it = tqdm(dataset.getIterator(self.config.BATCH_SIZE, task=current_task), disable=not self.verbose)
            it.set_description("Current task: {}".format(current_task))

            for x, y in it:
                x, y = x.to(self.config.DEVICE), \
                       y.to(self.config.DEVICE)

                with torch.no_grad():
                    y_pred_np = self.model.eval_forward(x).cpu().numpy()
                    y_true_np = y.cpu().numpy()

                y_true.extend(list(y_true_np))
                y_pred.extend(list(y_pred_np))
                i += 1

                it.set_postfix({'batch#': i})

            has_next_task = dataset.next_task(round_robin=False)

        return accuracy(y_true, y_pred), f1(y_true, y_pred)

    def load(self, task='last'):

        save_path = self.save_path+'_'+str(task)
        if not exists(save_path):
            if self.pretrained_model is not None:
                state = load(self.pretrained_model, map_location=self.device)

                ds = {n: k for n, k in state['model'].items() if n in self.model.state_dict().keys()
                      and k.size() == self.model.state_dict()[n].size()}

                for n, k in self.model.state_dict().items():
                    if n not in ds:
                        ds[n] = k

                self.model.load_state_dict(ds)

            return {}
        else:
            state = load(save_path, map_location=self.device)
            ds = {n: k for n, k in state['model'].items() if n in self.model.state_dict().keys()
                  and k.size() == self.model.state_dict()[n].size()}

            for n, k in self.model.state_dict().items():
                if n not in ds:
                    ds[n] = k

            self.model.load_state_dict(ds)

            results = {'metrics': state['metrics'],
                       'tasks': state['tasks'], 'losses': state['losses']}

            return results


if __name__ == '__main__':
    from utils.datasetsUtils.taskManager import SingleTargetClassificationTask, NoTask, IncrementalTaskClassification
    import utils.datasetsUtils.MINST as MINST
    import utils.datasetsUtils.CIFAR as CIFAR


    from networks import NoKafnet, Kafnet
    # import utils.datasetsUtils.CIFAR as CIFAR
    # import utils.datasetsUtils.MINST as MINST
    #
    # from utils.datasetsUtils.taskManager import SingleTargetClassificationTask, NoTask
    from networks.continual_learning import GEM, OnlineEWC, EWC
    from networks.continual_learning_beta import JaryGEM, embedding
    from configs.configClasses import DefaultConfig, OnlineLearningConfig
    from torchvision.transforms import transforms
    from copy import deepcopy
    from networks.net_utils import elu

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    dataset = CIFAR.Cifar100('./notebooks/data/cifar100', download=True, task_manager=IncrementalTaskClassification(10),
                            force_download=False, train_split=0.8, transform=transform, target_transform=None)
    dataset.load_dataset()

    d = deepcopy(dataset)

    # dataset = MINST.PermutedMINST('./data/minst', download=True, n_permutation=4,
    #                               force_download=False, train_split=0.8)
    # # dataset.load_dataset()
    #
    # # dataset = MINST.MINST('./data/minst', download=True, task_manager=NoTask,
    # #                       force_download=False, train_split=0.8, transform=None, target_transform=None)
    # dataset.load_dataset()
    #
    # # net = NoKafnet.MLP(len(dataset.class_to_idx), hidden_size=100)
    # net = Kafnet.KAFMLP(len(dataset.class_to_idx), hidden_size=int(400*0.7), kernel='gaussian', kaf_init_fcn=None)
    # # net = Kafnet.MultiKAFMLP(len(dataset.class_to_idx), hidden_size=int(400*0.7),# kaf_init_fcn=None,
    # #                          kernels=['gaussian'])

    # net = Kafnet.VGG(10, kernel='gaussian', D=10,  boundary=3, init_fcn=None, trainable_dict=True)

    config = DefaultConfig()

    # config.DEVICE = 'cpu'
    config.EPOCHS = 10
    config.IS_INCREMENTAL = True
    config.LR = 1e-1
    config.BATCH_SIZE = 32
    # config.EWC_IMPORTANCE = 0.5
    # config.EWC_SAMPLE_SIZE = 100
    # config.OPTIMIZER = 'Adam'
    config.CL_TEC = embedding
    config.USE_CL = True

    config.NEXT_TASK_LR = None
    config.NEXT_TASK_EPOCHS = None

    # config.CL_PAR = {'penalty_importance': 1, 'memorized_task_size': 300, 'weights_type': 'usage',
    #                  'sample_size': 50, 'maxf': 0.001, 'c': 2, 'margin': 0.5}

    config.CL_PAR = {'penalty_importance': 8, 'weights_type': 'distance', 'sample_size': 20, 'distance': 'cosine',
                     'supervised': True}

    net = NoKafnet.synCNN(100,  incremental=False)

    # net = Kafnet.CNN(10, kernel='gaussian', D=15, trainable_dict=False, boundary=4, topology=[int(32*0.85),  int(64*0.85)],
    #                  alpha_mean=0, alpha_std=0.8, init_fcn=None)

    # net = Kafnet.synCNN(10, kernel='softplus', D=10, boundary=3, trainable_dict=False,
    #                     topology=[int(32*0.9),  int(64*0.9)])

    print(sum([torch.numel(p) for p in net.parameters()]))

    for n, p in net.named_parameters():
        print(n, p.size())

    # net = NoKafnet.CNN(10)
    # net = Kafnet.VGG(10, kernel='gaussian', trainable_dict=False)

    print(config)
    trainer = Trainer(net, dataset, config)

    # print(trainer.model.state_dict().keys())
    # for k, v in trainer.model.state_dict().items():
    #     print(k, v.size())
    # input()
    #
    # trainer.load()
    # print(trainer.evaluate_on_dataset(dataset, 0, 0))

    a = trainer.all_tasks()
    # b = trainer.evaluate_on_dataset(d)
    # print(b)

    for k, v in a['tasks'].items():
        print(k, v['f1'])
        print('\t', v['accuracy'])

    print(a['metrics'])

    # dataset.load_dataset()
    # net = NoKafnet.VGG()

    # config.CL_TEC = OnlineEWC
    # config.USE_CL = True
    # # config.OPTIMIZER = 'Adam'
    #
    # # d = {'sample_size': 50, 'c': 1, 'weights_type': 'distance', 'external_embedding': (int(400*0.7), 100),
    # #      'memorized_task_size': 200}
    # #
    # # config.CL_PAR.update(d)
    #
    # # trainer = Trainer(net, dataset, config)
    # # a = trainer.all_tasks()
    # #
    # # for k, v in a['tasks'].items():
    # #     # print(k, v['accuracy'])
    # #     print(k, v['f1'])
    #
    # net = Kafnet.VGG()
    # dataset.load_dataset()
    # trainer = Trainer(net, dataset, config)
    # a = trainer.all_tasks()
    #
    # for k, v in a['tasks'].items():
    #     # print(k, v['accuracy'])
    #     print(k, v['f1'])

    # # config.USE_CL = False
    # #
    # # dataset.reset()
    # # net = NoKafnet.CNN(dataset.tasks_number)
    # # trainer = Trainer(net, dataset, config)
    # # a = trainer.all_tasks(2)
    # #
    # # print(a)
    # print(trainer.single_task())
