from model.BaseModel import BaseModel
from model.components.SimpleCNN import SimpleCNN
from model.utils.Progbar import Progbar
from model.utils.Config import Config
from model.utils.general import write_answers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


class MyModel(BaseModel):
    """Specialized class for My Model"""

    def __init__(self, config, dir_output):
        """
        Args:
            config: Config instance defining hyperparams

        """
        super(MyModel, self).__init__(config, dir_output)

    def build_train(self, config):
        """Builds model

        Args:
            config: from "training.json"

        """
        self.logger.info("- Building model...")

        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN().to(self.device)
        self._add_criterion(config.criterion_method)
        self._add_optimizer(config.lr_method, config.lr_init)

        self.logger.info("- done.")

    def build_pred(self):
        self.logger.info("- Building model...")

        # self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN()#.to(self.device)

        self.logger.info("- done.")

    def _run_train_epoch(self, config, train_set, val_set, epoch, lr_schedule):
        """Performs an epoch of training

        Args:
            config: Config instance
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            score: (float) model will select weights that achieve the highest
                score

        """
        # logging
        nbatches = len(train_set)
        prog = Progbar(nbatches)
        self.model.train()

        for i, (images, labels) in enumerate(train_set):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self._auto_backward(loss)

            prog.update(i + 1, [("loss", loss.item()), ("lr", lr_schedule.lr)])
            # update learning rate
            lr_schedule.update(batch_no=epoch*nbatches + i)

        # logging
        self.logger.info("- Training: {}".format(prog.info))

        self.model.eval()
        # evaluation
        config_eval = Config({"dir_answers": self._dir_output + "formulas_val/", "batch_size": config.batch_size})
        scores = self.evaluate(config_eval, val_set)
        score = scores["acc"]
        lr_schedule.update(score=score)

        return -score

    def _run_evaluate_epoch(self, config, test_set):
        """Performs an epoch of evaluation

        Args:
            test_set: Dataset instance
            params: (dict) with extra params in it
                - "dir_name": (string)

        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        with torch.no_grad():
            correct = 0
            total = 0
            preds = []
            refs = []
            for images, labels in test_set:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                for j in labels.tolist():
                    refs.append(j)
                correct += (predicted == labels).sum().item()
                pr = outputs[:, 1].detach().cpu().numpy()
                for i in pr:
                    preds.append(i)
            print('Test Accuracy {} %'.format(100 * correct / total))

        files = write_answers(refs, preds, config.dir_answers)

        return {
            "acc": 100 * correct / total
        }

    def predict_batch(self, images):
        preds = []
        images = images.to(self.device)
        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)
        pr = outputs[:, 1].detach().cpu().numpy()
        for i in pr:
            preds.append(i)

        return preds

    def predict(self, img):
        return self.predict_batch([img])
