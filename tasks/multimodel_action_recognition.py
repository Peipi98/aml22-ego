from abc import ABC
import torch
from utils import utils
from functools import reduce
import wandb
import tasks
from utils.logger import logger


class ActionRecognition(tasks.Task, ABC):
    loss = None
    optimizer = {}

    def __init__(self, name, task_models, batch_size, total_batch, models_dir, num_classes,
                 num_clips, model_args, args, **kwargs) -> None:
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        # Accuracy measures
        self.model_args = model_args
        self.accuracy = utils.Accuracy(topk=(1, 5), classes=num_classes)
        self.loss = utils.AverageMeter()
        self.num_clips = num_clips
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        optim_params = {}
        for m in self.modalities:
            if m == 'EMG':
                optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
                self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,
                                                momentum=model_args[m].sgd_momentum)
                continue

            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
            self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,
                                                weight_decay=model_args[m].weight_decay,
                                                momentum=model_args[m].sgd_momentum)

    def forward(self, data, **kwargs):
        # it is done for each modality, and then it is all saved in dicts
        logits = {}
        features = {}
        for i_m, m in enumerate(self.modalities):
            logits[m], feat = self.task_models[m](x=data[m], **kwargs)
            if i_m == 0:
                for k in feat.keys():
                    features[k] = {}
            for k in feat.keys():
                features[k][m] = feat[k]

        return logits, features

    def compute_loss(self, logits, label, loss_weight=1.0):
        # fuse all modalities together by summing the logits
        fused_logits = reduce(lambda x, y: x + y, logits.values())
        loss = self.criterion(fused_logits, label) / self.num_clips
        self.loss.update(torch.mean(loss_weight * loss) / (self.total_batch / self.batch_size), self.batch_size)

    def compute_accuracy(self, logits, label):
        # fuse all modalities together by summing the logits
        fused_logits = reduce(lambda x, y: x + y, logits.values())
        self.accuracy.update(fused_logits, label)

    def wandb_log(self):
        logs = {'loss verb': self.loss.val, 'top1-accuracy': self.accuracy.avg[1],
                'top5-accuracy': self.accuracy.avg[5]}
        for m in self.modalities:
            logs[f'lr_{m}'] = self.optimizer[m].param_groups[-1]['lr']
        wandb.log(logs)

    def reduce_learning_rate(self):
        for m in self.modalities:
            prev_lr = self.optimizer[m].param_groups[-1]["lr"]
            self.optimizer[m].param_groups[-1]["lr"] = self.optimizer[m].param_groups[-1]["lr"] / 10
            logger.info('Reducing learning rate modality {}: {} --> {}'
                        .format(m, prev_lr, self.optimizer[m].param_groups[-1]["lr"]))

    def reset_loss(self):
        self.loss.reset()

    def reset_acc(self):
        self.accuracy.reset()

    def step(self):
        super().step()
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph):
        self.loss.val.backward(retain_graph=retain_graph)
