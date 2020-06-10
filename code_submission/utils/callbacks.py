import numpy as np
from utils import get_logger

import copy

logger = get_logger("INFO")

BEST_VALID_TOP_NUM = 3


class Callback:
    def __init__(self): pass
    def on_train_begin(self, *args, **kwargs): pass
    def on_train_end(self, *args, **kwargs): pass
    def on_epoch_begin(self, *args, **kwargs): pass
    def on_epoch_end(self, *args, **kwargs): pass
    def on_batch_begin(self, *args, **kwargs): pass
    def on_batch_end(self, *args, **kwargs): pass
    def on_loss_begin(self, *args, **kwargs): pass
    def on_loss_end(self, *args, **kwargs): pass
    def on_step_begin(self, *args, **kwargs): pass
    def on_step_end(self, *args, **kwargs): pass


class EarlyStopping(Callback):
    def __init__(self, patience=5, tol=0.001, min_epochs=1, use_adaptive_topK=False):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.tol = tol
        self.best = -0.1
        # self.best = np.inf
        self.best_epoch = -1
        self.wait = 0
        self.stopped_epoch = -1
        # self.threshold = threshold
        self.min_epochs= min_epochs
        self.topK_list = []
        self.use_adaptive_topK = use_adaptive_topK
        self.loopn_best = {
            "pred": None,
            "acc": -1.0,
            "loss": 9999
        }

    def on_epoch_end(self, epoch, val_acc, epoch_loss, y_hat):
        use_adaptive_topK = self.use_adaptive_topK
        val_loss = min(1.0, val_acc + self.tol)
        if use_adaptive_topK:
            is_add = self.add_into_adaptive_topK(epoch, y_hat, val_acc, epoch_loss)
        else:
            self.topK_list = self.add_into_topK(self.topK_list, y_hat, val_acc, epoch_loss)

        if val_acc > self.best and self.best < 0.999:
            self.best = max(val_loss - self.tol, self.best)
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience and epoch > self.min_epochs:
                self.stopped_epoch = epoch
                logger.warning(
                    f"Early stopping conditioned on val_acc patience {self.patience} "
                    f"in epoch {self.stopped_epoch}. "
                    f"Metric is {val_acc}, best {self.best} in epoch {self.best_epoch}"
                )
                if use_adaptive_topK:
                    if is_add is False:
                        self.add_into_adaptive_topK(epoch, y_hat, val_acc, epoch_loss, early_stop=True)
                return True
        return False

    def add_into_topK(self, topK_list, y_hat, acc, loss):
        valid_dict = {
            "pred": None,
            "acc": acc,
            "loss": loss
        }

        if len(topK_list) < BEST_VALID_TOP_NUM:
            valid_dict["pred"] = y_hat
            topK_list.append(valid_dict)
            return topK_list
        if (acc <= topK_list[-1]["acc"]) or ((acc == topK_list[-1]["acc"]) and (loss > topK_list[-1]["loss"])):
            return topK_list
        valid_dict["pred"] = y_hat
        topK_list[-1] = valid_dict
        topK_list = sorted(topK_list, key=lambda x: (-x["acc"], x["loss"]))
        return topK_list

    def add_into_adaptive_topK(self, epoch, y_hat, acc, loss, early_stop=False):
        if early_stop or (epoch > 0 and (epoch + 1) % 10 == 0):
            # self.topK_list.append(self.loopn_best)
            self.topK_list = self.add_into_topK(self.topK_list, self.loopn_best["pred"], self.loopn_best["acc"], self.loopn_best["loss"])
            self.loopn_best = {
                "pred": None,
                "acc": -1.0,
                "loss": 9999
            }
            return True
        else:
            valid_dict = {
                "pred": None,
                "acc": acc,
                "loss": loss
            }
            if (acc > self.loopn_best["acc"]) or ((acc == self.loopn_best["acc"]) and (loss < self.loopn_best["loss"])):
                valid_dict["pred"] = y_hat
                self.loopn_best = valid_dict
            return False
