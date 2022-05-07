import numpy as np
import torch
import wandb

from callback.callback import Callback
from sklearn import metrics


class WandbCallback(Callback):
    def __init__(self) -> None:
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.train_batch_losses = []
        self.val_batch_losses = []

    def on_val_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        gts = gts.detach().cpu()
        preds = preds.detach().cpu()

        fpr, tpr, threshold = metrics.roc_curve(gts, preds)
        roc_auc = metrics.auc(fpr, tpr)
        wandb.log({"roc_auc": roc_auc})

        print(f"roc_auc: {roc_auc}")
        return True

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        self.train_batch_losses.append(loss)

    def on_epoch_end(self, loss, val_loss, model: torch.nn.Module) -> bool:
        self.val_epoch_losses.append(val_loss)
        self.train_epoch_losses.append(loss)

        wandb.log({"loss": loss, "val_loss": val_loss})

        return True
