from model.BaseModel import BaseModel
from model.components.SimpleCNN import SimpleCNN
from model.components.ResNet import ResNet9
from model.utils.Progbar import Progbar
from model.utils.Config import Config
from model.utils.general import write_answers
import numpy as np
import torch


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
        if config.model == "CNN":
            self.model = SimpleCNN()
        else:
            self.model = ResNet9()
        self.model = self.model.to(self.device)
        self._add_criterion(config.criterion_method)
        self._add_optimizer(config.lr_method, config.lr_init)

        self.logger.info("- done.")

    def build_pred(self, config):
        self.logger.info("- Building model...")

        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        if config.model == "CNN":
            self.model = SimpleCNN()
        else:
            self.model = ResNet9()
        self.model = self.model.to(self.device)

        self.logger.info("- done.")

    def _run_train_epoch(self, config, train_set, val_set, epoch, lr_schedule, path_label):
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
            if config.model == "CNN":
                loss = self.criterion(outputs, labels)
            else:
                loss = self.criterion(outputs, labels.view(-1, 1).type(torch.FloatTensor).to(self.device))
            self._auto_backward(loss)

            # scheduler.step()
            # scheduler.get_lr
            # 衰减学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_schedule.lr

            prog.update(i + 1, [("loss", loss.item()), ("lr", lr_schedule.lr)])
            # update learning rate
            lr_schedule.update(batch_no=epoch*nbatches + i)

        # logging
        self.logger.info("- Training: {}".format(prog.info))

        # evaluation
        config_eval = Config({
            "dir_answers": self._dir_output + "formulas_val/",
            "batch_size": config.batch_size,
            "model": config.model
        })
        scores = self.evaluate(config_eval, val_set, path_label)
        score = scores["acc"]
        lr_schedule.update(score=score)

        return score

    def _run_evaluate_epoch(self, config, test_set, path_label):
        """Performs an epoch of evaluation

        Args:
            test_set: Dataset instance
            params: (dict) with extra params in it
                - "dir_name": (string)

        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        self.model.eval()
        with torch.no_grad():
            nbatches = len(test_set)
            prog = Progbar(nbatches)
            correct = 0
            total = 0
            preds = []
            refs = []
            for k, (images, labels) in enumerate(test_set):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)

                for j in labels.tolist():
                    refs.append(j)
                if config.model == "CNN":
                    pr = outputs[:, 1].detach().cpu().numpy()
                    for i in pr:
                        preds.append(1 if i > 0 else 0)
                else:
                    pr = outputs[:].detach().cpu().numpy()
                    for i in pr:
                        preds.append(1 if i[0] > 0 else 0)
                total = len(refs)
                print(np.asarray(refs) == np.asarray(preds))
                print(len((np.asarray(refs) == np.asarray(preds)).tolist()))
                print(refs[:5])
                print(preds[:5])
                correct = (np.asarray(refs) == np.asarray(preds)).sum().item()
                prog.update(k + 1, [("acc", correct / total), ("correct", correct), ("total", total)])
        self.logger.info("- Evaluating: {}".format(prog.info))
        write_answers(refs, preds, config.dir_answers, path_label)

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
