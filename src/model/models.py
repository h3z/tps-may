import torch
from model import mlp, small_mlp, small_mlp2
import wandb


def get(input_size) -> torch.nn.Module:
    if wandb.config["model"] == "baseline":
        return small_mlp.Model(input_size).to("cuda")
