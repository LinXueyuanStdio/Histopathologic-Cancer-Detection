import os
import sys
import time

from .utils.general import init_dir, get_logger
import torch.nn as nn
import torch


class BaseModel(object):
    """Generic class for tf models"""

    def __init__(self, config, dir_output):
        """Defines self._config

        Args:
            config: (Config instance) class with hyper parameters, from "model.json"

        """
        self._config = config
        self._dir_output = dir_output
        self._init_relative_path(dir_output)
        self.logger = get_logger(dir_output + "model.log")

    def _init_relative_path(self, dir_output):
        # init parent dir
        init_dir(dir_output)

        # 1. init child dir
        # check dir one last time
        self._dir_model = dir_output + "model_weights/"
        init_dir(self._dir_model)
        self._model_path = self._dir_model+"model.cpkt"

    def build_train(self, config=None):
        """To overwrite with model-specific logic

        This logic must define
            - self.model
            - self.loss
            - self.lr
            - etc.
        """
        raise NotImplementedError

    def build_pred(self, config=None):
        """Similar to build_train but no need to define train_op"""
        raise NotImplementedError

    def _add_optimizer(self, lr_method, lr):
        """Defines self.optimizer that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: learning rate

        """
        _lr_m = lr_method.lower()  # lower to make sure
        print("  - " + lr_method)
        if _lr_m == 'adam':  # sgd method
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif _lr_m == 'adamax':
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=lr)
        elif _lr_m == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Unknown method {}".format(_lr_m))

        print("  - lr_scheduler.CosineAnnealingLR")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=4e-08)

    def _add_criterion(self, criterion_method):
        """Defines self.criterion that performs an update on a batch

        Args:
            criterion_method: (string) criterion method, for example "CrossEntropyLoss"

        """
        _criterion_method = criterion_method
        print("  - " + criterion_method)

        if _criterion_method == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif _criterion_method == 'MSELoss':
            self.criterion = torch.nn.MSELoss()
        elif _criterion_method == 'BCEWithLogitsLoss':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("Unknown method {}".format(_criterion_method))

    def _auto_backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def auto_restore(self):
        if os.path.exists(self._model_path) and os.path.isfile(self._model_path):
            self.restore()

    def restore(self, model_path=None, map_location='cpu'):
        """Reload weights into session

        Args:
            sess: tf.Session()
            model_path: weights path "model_weights/model.cpkt"

        """
        self.logger.info("- Reloading the latest trained model...")
        if model_path == None:
            self.model.load_state_dict(torch.load(self._model_path, map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=map_location))

    def save(self):
        """Saves model"""
        self.logger.info("- Saving model...")
        torch.save(self.model.state_dict(), self._model_path)
        self.logger.info("- Saved model in {}".format(self._dir_model))

    def train(self, config, train_set, val_set, lr_schedule, path_label):
        """Global training procedure

        Calls method self.run_epoch and saves weights if score improves.
        All the epoch-logic including the lr_schedule update must be done in
        self.run_epoch

        Args:
            config: Config instance contains params as attributes
            train_set: Dataset instance
            val_set: Dataset instance
            lr_schedule: LRSchedule instance that takes care of learning proc
            path_label: dataframe

        Returns:
            best_score: (float)

        """
        best_score = None

        for epoch in range(config.n_epochs):
            # logging
            tic = time.time()
            self.logger.info("Epoch {:}/{:}".format(epoch+1, config.n_epochs))

            # epoch
            score = self._run_train_epoch(config, train_set, val_set, epoch, lr_schedule, path_label)

            # save weights if we have new best score on eval
            if best_score is None or score >= best_score:
                best_score = score
                self.logger.info("- New best score ({:04.2f})!".format(best_score))
                self.save()
            if lr_schedule.stop_training:
                self.logger.info("- Early Stopping.")
                break

            # logging
            toc = time.time()
            self.logger.info("- Elapsed time: {:04.2f}, learning rate: {:04.5f}".format(toc-tic, lr_schedule.lr))

        return best_score

    def _run_train_epoch(config, train_set, val_set, epoch, lr_schedule, path_label):
        """Model_specific method to overwrite

        Performs an epoch of training

        Args:
            config: Config
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            score: (float) model will select weights that achieve the highest
                score

        """
        raise NotImplementedError

    def evaluate(self, config, test_set, path_label):
        """Evaluates model on test set

        Calls method run_evaluate on test_set and takes care of logging

        Args:
            config: Config
            test_set: instance of class Dataset
            path_label: dataframe

        Return:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        self.logger.info("- Evaluating...")
        scores = self._run_evaluate_epoch(config, test_set, path_label)  # evaluate
        msg = " ... ".join([" {} is {:04.2f} ".format(k, v) for k, v in scores.items()])
        self.logger.info("- Eval: {}".format(msg))

        return scores

    def _run_evaluate_epoch(config, test_set):
        """Model-specific method to overwrite

        Performs an epoch of evaluation

        Args:
            config: Config
            test_set: Dataset instance

        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        raise NotImplementedError
