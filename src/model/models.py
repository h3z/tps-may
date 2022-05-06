import torch
from model import mlp


def get() -> torch.nn.Module:
    return mlp.Model().to("cuda")
