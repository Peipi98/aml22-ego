import os
from datetime import datetime
from pathlib import Path
import torch
from abc import ABCMeta, abstractmethod
from utils.logger import logger

"""
Task is the abstract class which needs to be implemented for every different task present in the model 
(i.e. classification, self-supervised). It saves all models for every modality.
"""


class Task(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, name, task_models, batch_size, total_batch, models_dir, args, **kwargs) -> None:
        """
        name: str, name of the task e.g. action_classifier, domain_classifier...
        task_models: dict(str, Net), containing one model for each different modality adopted by the task
        batch_size: actual batch size in the forward
        total_batch: batch size simulated via gradient accumulation
        models_dir: directory where the models are stored when saved
        """
        super().__init__()
        self.name = name
        self.task_models = task_models
        self.modalities = list(self.task_models.keys())
        self.batch_size = batch_size
        self.total_batch = total_batch
        self.models_dir = models_dir
        self.current_iter = 0
        self.best_iter = 0
        self.best_iter_score = 0
        self.last_iter_acc = 0
        self.model_count = 1
        self.args = args

        self.kwargs = kwargs

    @property
    @abstractmethod
    def optimizer(self):
        """
        to force subclasses to implement their own optimizers
        """
        pass

    @property
    @abstractmethod
    def loss(self):
        """
        to force subclasses to set a loss
        """
        pass

    def load_on_gpu(self, device=torch.device('cuda')):
        """
        function to load all the models related to the task on the different GPUs used
        """
        for modality, model in self.task_models.items():
            self.task_models[modality] = torch.nn.DataParallel(model).to(device)

    def load_model(self, path, idx):
        """
        Function to load a specific model (idx-one) among the last 9 saved from a specific path,
        might be overwritten in case the task requires it
        path: path-like string containing the path to the model
        idx: int, identifier of the model to be restored
        """
        # list all files in chronological order (1st is most recent, last is less recent)
        last_dir = Path(list(sorted(Path(path).iterdir(), key=lambda date: datetime.strptime(
            os.path.basename(os.path.normpath(date)), "%b%d_%H-%M-%S")))[-1])
        last_models_dir = last_dir.iterdir()
        for m in self.modalities:
            #In this case we just upload the model without using the EK conf
            if m == 'EMG':
                model_path = os.path.join('./pretrained_emgcnn/', 'EMG_CNN.pth')
                torch.load(model_path)
                continue

            # get only models which belong to this task and for this modality
            model = list(filter(lambda x:
                                m == x.name.split('.')[0].split('_')[-2] and
                                self.name == x.name.split('.')[0].split('_')[-3] and
                                str(idx) == x.name.split('.')[0].split('_')[-1], last_models_dir))[0].name
            model_path = os.path.join(str(last_dir), model)
            logger.info('Restoring {} for modality {} from {}'.format(self.name, m, model_path))
            checkpoint = torch.load(model_path)

            self.current_iter = checkpoint['iteration']
            self.best_iter = checkpoint['best_iter']
            self.best_iter_score = checkpoint['best_iter_score']
            self.last_iter_acc = checkpoint['acc_mean']
            self.task_models[m].load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.optimizer[m].load_state_dict(checkpoint['optimizer_state_dict'])
            try:
                self.model_count = checkpoint['last_model_count_saved']
                self.model_count = self.model_count + 1 if self.model_count < 9 else 1
            except KeyError:
                # for compatibility with models saved before refactoring
                self.model_count = 1

            logger.info("{}-Model for {} restored at iter {}\n"
                        "Best accuracy on val: {:.2f} at iter {}\n"
                        "Last accuracy on val: {:.2f}\n"
                        "Last loss: {:.2f}".format(m, self.name, self.current_iter, self.best_iter_score,
                                                   self.best_iter, self.last_iter_acc, checkpoint['loss_mean']))

    def load_last_model(self, path):
        """
        Function to load the last model from a specific path, might be overwritten in case the task requires it
        path: path-like string containing the path to the model
        """
        # list all files in chronological order (1st is most recent, last is less recent)
        last_models_dir = list(sorted(Path(path).iterdir(), key=lambda date: datetime.strptime(
            os.path.basename(os.path.normpath(date)), "%b%d_%H-%M-%S")))[-1]
        saved_models = [x for x in reversed(sorted(Path(last_models_dir).iterdir(), key=os.path.getmtime))]
        for m in self.modalities:
            if m == 'EMG':
                model_path = os.path.join('./pretrained_emgcnn/', 'EMG_CNN.pth')
                torch.load(model_path)
                continue
            # get only models which belong to this task and for this modality
            model = list(filter(lambda x:
                                m == x.name.split('.')[0].split('_')[-2] and
                                self.name == x.name.split('.')[0].split('_')[-3], saved_models))[0].name
            model_path = os.path.join(last_models_dir, model)
            logger.info('Restoring {} for modality {} from {}'.format(self.name, m, model_path))
            checkpoint = torch.load(model_path)

            self.current_iter = checkpoint['iteration']
            self.best_iter = checkpoint['best_iter']
            self.best_iter_score = checkpoint['best_iter_score']
            self.last_iter_acc = checkpoint['acc_mean']
            self.task_models[m].load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.optimizer[m].load_state_dict(checkpoint['optimizer_state_dict'])
            try:
                self.model_count = checkpoint['last_model_count_saved']
                self.model_count = self.model_count + 1 if self.model_count < 9 else 1
            except KeyError:
                # for compatibility with models saved before refactoring
                self.model_count = 1

            logger.info("{}-Model for {} restored at iter {}\n"
                        "Best accuracy on val: {:.2f} at iter {}\n"
                        "Last accuracy on val: {:.2f}\n"
                        "Last loss: {:.2f}".format(m, self.name, self.current_iter, self.best_iter_score,
                                                   self.best_iter, self.last_iter_acc, checkpoint['loss_mean']))

    def save_model(self, current_iter, last_iter_acc, prefix=None):
        """
        Function to save the model, might be overwritten in case the task requires it
        current_iter: int, current iteration in which the model is going to be saved
        last_iter_accuracy: float, the accuracy reached in the last iteration
        count: int, number of model saved (i.e. there is a limited number of models to be saved, count is just a
               parameter to differentiate on those)
        prefix: str, string to be put as a prefix to filename of the model to be saved
        """
        for m in self.modalities:
            if prefix is not None:
                filename = prefix + '_' + self.name + '_' + m + '_' + str(self.model_count) + '.pth'
            else:
                filename = self.name + '_' + m + '_' + str(self.model_count) + '.pth'
            if not os.path.exists(os.path.join(self.models_dir, self.args.experiment_dir)):
                os.makedirs(os.path.join(self.models_dir, self.args.experiment_dir))
            try:
                torch.save({'iteration': current_iter,
                            'best_iter': self.best_iter,
                            'best_iter_score': self.best_iter_score,
                            'acc_mean': last_iter_acc,
                            'loss_mean': self.loss.acc,
                            'model_state_dict': self.task_models[m].state_dict(),
                            'optimizer_state_dict': self.optimizer[m].state_dict(),
                            'last_model_count_saved': self.model_count
                            }, os.path.join(self.models_dir, self.args.experiment_dir, filename))
                self.model_count = self.model_count + 1 if self.model_count < 9 else 1

            except Exception as e:
                logger.info("An error occurred while saving the checkpoint:")
                logger.info(e)

    def train(self, mode=True):
        """
        activate the training in all models (when training, DropOut is active, BatchNorm updates itself)
        (when not training, BatchNorm is freezed, DropOut disabled)
        """
        for model in self.task_models.values():
            model.train(mode)

    def zero_grad(self):
        """
        reset the gradient when gradient accumulation is finished
        """
        for m in self.modalities:
            self.optimizer[m].zero_grad()

    def step(self):
        """
        perform the optimization step once all the gradients of the gradient accumulation are accumulated
        """
        for m in self.modalities:
            self.optimizer[m].step()

    def check_grad(self):
        """
        check that the gradients of the model are not over a certain threshold
        """
        for m in self.modalities:
            for name, param in self.task_models[m].named_parameters():
                if param.requires_grad and param.grad is not None:
                    if param.grad.norm(2).item() > 25:
                        logger.info(f"Param {name} has a gradient whose L2 norm is over 25")

    def __str__(self) -> str:
        return self.name
