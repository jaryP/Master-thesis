from tqdm import tqdm
import configs.configClasses as configClasses
from utils.datasetsUtils.dataset import GeneralDatasetLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import optim, sum, abs, save, load
from utils.metrics import MetricsHolder
from networks.net_utils import AbstractNetwork
from os.path import join, exists
from os import makedirs
import warnings


class Trainer:
    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config: configClasses.DefaultConfig,
                 save_modality=1):

        self.config = config
        self.dataset = dataset
        self.model = model
        self.metrics_calculator = MetricsHolder(self.dataset.tasks_number)

        self.save_modality = save_modality
        self.device = config.DEVICE

        self.save_path = join(config.SAVE_PATH, config.MODEL_NAME)
        if not exists(config.SAVE_PATH):
            makedirs(config.SAVE_PATH)

        if config.DEVICE != 'cpu':
            self.model.to(self.device)

        self.ewc = None
        if config.USE_EWC:
            if config.EWC_TYPE is None:
                self.ewc = None
                warnings.warn("Ewc type is set to None  ")
            else:
                self.ewc = config.EWC_TYPE(self.model, self.dataset, config)

        self.results = dict()

        if config.LOSS == 'cross_entropy':
            self.loss_function = F.cross_entropy
        else:
            raise ValueError('Not known loss function, allowed ones are: cross_entropy')

        if config.OPTIMIZER == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.LR)
        else:
            raise ValueError('Not known optimizer, allowed ones are: SGD')

    def single_task(self, task=0):
        losses = []

        self.dataset.task = task

        for e in range(self.config.EPOCHS):
            loss = self.epoch(e)
            losses.append(loss)
            self.evaluate()

        if self.save_modality >= 1:
            state = {'metrics': self.metrics_calculator.metrics, 'losses': losses,
                     'model': self.model.state_dict()}
            save(state, self.save_path + "_last")

        return {'losses': losses, 'metrics': self.metrics_calculator.metrics}

    def all_tasks(self, limit=-1):

        has_next_task = True
        losses_per_task = dict()

        while has_next_task:

            current_task = self.dataset.task

            if 0 < limit <= current_task:
                break

            losses = []

            # if (current_task > 0) and self.ewc is not None:
            #     self.ewc(current_task=current_task)

            self.dataset.task = current_task
            self.model.task = current_task

            for e in range(self.config.EPOCHS):
                loss = self.epoch(e)
                losses.append(loss)

                self.evaluate()

                if current_task > 0:
                    for sub_task in range(current_task):
                        self.evaluate(sub_task)

                self.dataset.task = current_task

            for sub_task in range(current_task+1, self.dataset.tasks_number):
                self.evaluate(sub_task)

            losses_per_task[current_task] = losses
            has_next_task = self.dataset.next_task(round_robin=False)

            # a = self.metrics_calculator.metrics
            # for k, v in a['tasks'].items():
            #     print(k, v['accuracy'])

            if self.save_modality == 2:
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
                     'model': state_dict}

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

        it = tqdm(self.dataset.getIterator(self.config.BATCH_SIZE), total=len(self.dataset)//self.config.BATCH_SIZE)

        if self.ewc is not None:
            it.set_description("Training task (ewc) {}, epoch {}".format(self.dataset.task, n+1))
        else:
            it.set_description("Training task {}, epoch {}".format(self.dataset.task, n+1))

        loss_calcualted = False

        for x, y in it:

            self.model.task = current_task

            x, y = x.to(self.config.DEVICE), \
                            y.to(self.config.DEVICE)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss_function(output, y)

            if self.config.L1_REG > 0:
                l1_loss = 0.0
                for name, param in self.model.named_parameters():
                    l1_loss += sum(abs(param))
                loss = loss + self.config.L1_REG * l1_loss

            loss.backward(retain_graph=True)

            if self.ewc is not None:
                self.ewc, penality = self.ewc(current_task=self.dataset.task)

                if penality != 0:
                    loss = loss + self.config.EWC_IMPORTANCE * penality.to(self.config.DEVICE)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)

            clip_grad_norm_(self.model.parameters(), 20)
            self.optimizer.step()

            epoch_loss_full += loss.detach().item()
            i += 1

            if self.ewc is None:
                it.set_postfix({'loss': epoch_loss_full/i,
                                'batch#': i})
            else:
                it.set_postfix({'loss': epoch_loss_full / i, 'batch#': i})

        return epoch_loss_full/i

    def evaluate(self, evaluated_task=None):
        self.dataset.test_phase()
        self.model.eval()
        self.model.task = evaluated_task if evaluated_task is not None else self.dataset.task

        i = 0

        it = tqdm(self.dataset.getIterator(self.config.BATCH_SIZE, task=evaluated_task))
        it.set_description("Testing task {}".format(self.dataset.task if evaluated_task is None else evaluated_task))

        y_true = []
        y_pred = []

        for x, y in it:
            x, y = x.to(self.config.DEVICE), \
                   y.to(self.config.DEVICE)

            y_pred_np = self.model.eval_forward(x)
            y_true_np = y.cpu().detach().numpy()

            y_true.extend(y_true_np)
            y_pred.extend(y_pred_np)
            i += 1

            it.set_postfix({'batch#': i})

        self.metrics_calculator.add_evaluation(evaluated_task, self.dataset.task, y_true=y_true, y_pred=y_pred)

    def load(self, task='last'):
        save_path = self.save_path+'_'+task
        if not exists(save_path):
            return {}
        else:
            state = load(save_path, map_location=self.device)

            self.model.load_state_dict(state['model'])
            results = {'metrics': state['metrics'], 'tasks': state['tasks'], 'losses': state['losses']}
            return results


if __name__ == '__main__':
    from networks import NoKafnet, Kafnet
    import utils.datasetsUtils.CIFAR as CIFAR
    import utils.datasetsUtils.MINST as MINST

    from utils.datasetsUtils.taskManager import SingleTargetClassificationTask, NoTask
    from networks.continual_learning import GEM, Bayesian
    from configs.configClasses import DefaultConfig, OnlineLearningConfig
    from torchvision.transforms import transforms


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    dataset = MINST.PermutedMINST('./data/minst', download=True, n_permutation=4,
                                  force_download=False, train_split=0.8)
    dataset.load_dataset()

    # net = NoKafnet.MLP(len(dataset.class_to_idx))
    # net = Kafnet.KAFMLP(len(dataset.class_to_idx), hidden_size=int(400), kernel='gaussian', kaf_init_fcn=None)
    net = Kafnet.MultiKAFMLP(len(dataset.class_to_idx), hidden_size=int(400-0.7), kaf_init_fcn=None, kernel_combination='softmax')

    config = OnlineLearningConfig()
    config.EPOCHS = 5
    config.LR = 2e-3
    config.EWC_IMPORTANCE = 1000
    config.EWC_SAMPLE_SIZE = 200
    config.EWC_TYPE = OnlineLearningConfig.EWC_TYPE

    config.USE_EWC = True
    config.EWC_IMPORTANCE = 1000
    config.L1_REG = 0
    config.IS_CONVOLUTIONAL = False
    print(config)

    trainer = Trainer(net, dataset, config)
    a = trainer.all_tasks()
    print(a['tasks'])
    for k, v in a['tasks'].items():
        print(k, v['accuracy'])

    # config.USE_EWC = False
    #
    # dataset.reset()
    # net = NoKafnet.CNN(dataset.tasks_number)
    # trainer = Trainer(net, dataset, config)
    # a = trainer.all_tasks(2)
    #
    # print(a)
    # print(trainer.single_task())
