import wandb, utils, os, pickle
from model import models
from data import data_split, data_process, data_loader, data_reader
from train import train, losses, optimizers, schedulers
from callback import early_stopping, wandb_callback

from config import config
from datetime import datetime

utils.fix_random()


def get_parameters():
    return {
        "~lr": 0.01,
        "~batch_size": 4096,
        "~epochs": 60,
        "~early_stopping_patience": 6,
        "~optimizer": "adam",
        "~loss": "bce",
        "activation": "swish",  # relu, swish
        "model": "baseline",  # baseline, rm_batchnorm
    }


class TMP:
    def __init__(self) -> None:
        self.aaa = datetime.now()

    def ppp(self, i=None):
        if i:
            print(i, datetime.now() - self.aaa)
        self.aaa = datetime.now()


tmp = TMP()


def main():
    tmp.ppp(1)
    wandb.init(config=get_parameters(), **config.__wandb__)
    print(wandb.config)
    tmp.ppp("wandb init")

    # read csv
    dr = data_reader.DataReader()
    df = dr.train

    # split
    train_df, val_df, _ = data_split.split(df)
    test_df = dr.test

    # preprocess
    tmp.ppp()
    processor = data_process.DataProcess(train_df)
    train_df = processor.preprocess(train_df)
    val_df = processor.preprocess(val_df)
    test_df = processor.preprocess(test_df)
    tmp.ppp("preprocess")

    # torch DataLoader
    train_ds = data_loader.DataLoader(train_df).get(is_train=True)
    val_ds = data_loader.DataLoader(val_df).get()
    test_ds = data_loader.DataLoader(test_df).get()

    model = models.get(len(set(train_df.columns) - {"id", "target"}))
    tmp.ppp("model to cuda")

    # train
    criterion = losses.get()
    optimizer = optimizers.get(model)
    scheduler = schedulers.get(optimizer, train_ds)
    # scheduler = None
    callbacks = [early_stopping.EarlyStopping(), wandb_callback.WandbCallback()]

    for epoch in range(wandb.config["~epochs"]):
        loss = train.epoch_train(
            model, optimizer, scheduler, train_ds, criterion, callbacks
        )
        val_loss = train.epoch_val(model, val_ds, criterion, callbacks)
        print(epoch, ": train_loss", loss, "val_loss", val_loss)

        res = [c.on_epoch_end(loss, val_loss, model) for c in callbacks]
        if False in res:
            print("Early stopping")
            break

    [c.on_train_finish(model) for c in callbacks]

    # predict
    preds, gts = train.predict(model, test_ds)

    # post process
    preds, gts = processor.postprocess(preds), processor.postprocess(gts)

    dr.submit(preds.squeeze())

    wandb.finish()
    tmp.ppp("final")


if __name__ == "__main__":
    main()
