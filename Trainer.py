from tqdm import tqdm

from configs import configs
from networks.continual_learning import EWC
from utils.datasetsUtils.dataset import GeneralDatasetLoader
import torch.nn.functional as F
from torch import nn, optim, sum, abs
import random
from utils.metrics import calculate_all_metrics


class Trainer:
    def __init__(self, model: nn.Module, dataset: GeneralDatasetLoader, config: configs.DefaultConfig):

        self.config = config
        self.dataset = dataset
        self.model = model

        if config.USE_EWC:
            self.ewc = EWC(self.model, self.dataset, config)
        else:
            self.ewc = None

        if config.DEVICE != 'cpu':
            self.model.cuda(config.DEVICE)

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
        metrics = []

        self.dataset.task = task

        for e in range(self.config.EPOCHS):
            loss = self.epoch(e)
            losses.append(loss)

            m = self.evaluate()
            metrics.append(m)

        return losses, metrics

    def all_tasks(self, limit=-1):

        has_next_task = True
        results = dict()
        current_task = 0

        while has_next_task:

            current_task = self.dataset.task

            if limit > 0 and current_task >= limit:
                break

            losses = []
            metrics = []

            if (current_task > 0) and self.ewc is not None:
                # If we want EWC and it is at least the second task, sample the previous datasetsUtils
                old_tasks = []
                for sub_task in range(current_task):

                    self.dataset.task = sub_task
                    self.dataset.train_phase()
                    it = self.dataset.getIterator(self.config.EWC_SAMPLE_SIZE)

                    images, _ = next(it)

                    old_tasks.extend([(images[i], sub_task) for i in range(len(images))])

                old_tasks = random.sample(old_tasks, k=self.config.EWC_SAMPLE_SIZE)
                self.ewc = self.ewc(old_tasks=old_tasks)

            self.dataset.task = current_task

            for e in range(self.config.EPOCHS):
                loss = self.epoch(e)
                losses.append(loss)

                m = self.evaluate()
                metrics.append(m)

                if current_task > 0:
                    for sub_task in range(current_task):
                        self.dataset.task = sub_task
                        m = self.evaluate()
                        r = results[sub_task]
                        m_old = r['metrics']
                        m_old.append(m)
                        results[sub_task]['metrics'] = m_old

                self.dataset.task = current_task

            results[current_task] = {'losses': losses, 'metrics': metrics}

            has_next_task = self.dataset.next_task(round_robin=False)

        return results

    def epoch(self, n):
        self.model.train()
        self.dataset.train_phase()

        i = 0
        epoch_loss_full = 0

        it = tqdm(self.dataset.getIterator(self.config.BATCH_SIZE), total=len(self.dataset)//self.config.BATCH_SIZE)
        it.set_description("Training task {}, epoch {}".format(self.dataset.task, n+1))

        for x, y in it:
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

            if self.ewc is not None:
                penality = self.ewc.penalty(self.model)
                loss = loss + self.config.EWC_IMPORTANCE * penality

            loss.backward()
            self.optimizer.step()

            epoch_loss_full += loss.detach().item()
            i += 1

            it.set_postfix({'loss': epoch_loss_full/i,
                            'batch#': i})

        return epoch_loss_full/i

    def evaluate(self):
        self.dataset.test_phase()
        self.model.eval()

        i = 0

        it = tqdm(self.dataset.getIterator(self.config.BATCH_SIZE), total=len(self.dataset)//self.config.BATCH_SIZE)
        it.set_description("Testing task {}".format(self.dataset.task))

        y_true = []
        y_pred = []

        for x, y in it:
            x, y = x.to(self.config.DEVICE), \
                   y.to(self.config.DEVICE)

            y_pred_np = self.model.eval_forward(self.model(x))
            y_true_np = y.cpu().detach().numpy()

            y_true.extend(y_true_np)
            y_pred.extend(y_pred_np)
            i += 1

            it.set_postfix({'batch#': i})

        return calculate_all_metrics(y_true, y_pred)


if __name__ == '__main__':
    from networks import NoKafnet
    import utils.datasetsUtils.CIFAR as CIFAR
    from utils.datasetsUtils.taskManager import SingleTargetClassificationTask, NoTask
    from configs.configs import DefaultConfig
    from torchvision.transforms import transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    dataset = CIFAR.Cifar10('./data/cifar10', SingleTargetClassificationTask(), download=True,
                            force_download=False, train_split=0.8, transform=transform, target_transform=None)
    dataset.load_dataset()

    net = NoKafnet.CNN(dataset.tasks_number)
    config = DefaultConfig()
    config.EPOCHS = 1
    config.L1_REG = 0

    trainer = Trainer(net, dataset, config)
    a = trainer.all_tasks(3)
    print(a)

    # config.USE_EWC = False
    #
    # dataset.reset()
    # net = NoKafnet.CNN(dataset.tasks_number)
    # trainer = Trainer(net, dataset, config)
    # a = trainer.all_tasks(2)
    #
    # print(a)
    # print(trainer.single_task())
