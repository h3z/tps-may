import torch
from torch import nn
import torch.nn.functional as F
import wandb


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 1)
        if wandb.config["activation"] == "relu":
            self.activation = F.relu
        elif wandb.config["activation"] == "swish":
            self.activation = F.silu

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn5(self.fc5(x)))
        x = torch.sigmoid(self.fc6(x))

        return x.squeeze()
