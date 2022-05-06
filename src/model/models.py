import torch
from model import mlp


def get(input_size) -> torch.nn.Module:
    return mlp.Model(input_size).to("cuda")
