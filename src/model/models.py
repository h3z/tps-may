import torch
from model import mlp, small_mlp, small_mlp2


def get(input_size) -> torch.nn.Module:
    return small_mlp.Model(input_size).to("cuda")
