import torch
import random
from config.config import RANDOM_STATE


def fix_random():
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
