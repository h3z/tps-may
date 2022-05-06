import torch
import wandb


def get():
    if wandb.config["~loss"] == "mse":
        return torch.nn.MSELoss()
    elif wandb.config["~loss"] == "sigmoid":
        return torch.nn.BCELoss()
