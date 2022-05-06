import pandas as pd
import numpy as np
from typing import List
import torch
import random
import wandb


class DataLoader:
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray):
            self.x = x
            self.y = y
            self.len = len(self.x)

        def __getitem__(self, index):
            x = self.x[index]
            y = self.y[index]
            return x, y

        def __len__(self):
            return self.len

    class Sampler(torch.utils.data.Sampler):
        def __init__(self, l: int, shuffle: bool) -> None:
            super().__init__(l)
            self.len = l
            self.shuffle = shuffle

        def __iter__(self) -> List[int]:
            lst = list(range(self.len))
            if self.shuffle:
                random.shuffle(lst)
            for i in lst:
                yield i

        def __len__(self) -> int:
            return self.len

    def __init__(self, df: pd.DataFrame) -> None:
        self.x = df.drop(columns=["id", "target"]).values
        self.y = df["target"].values

    def get(self, is_train=False) -> torch.utils.data.DataLoader:
        dataset = self.Dataset(self.x, self.y)
        sampler = self.Sampler(len(self.x), shuffle=is_train)
        batch_size = wandb.config["~batch_size"] if is_train else len(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=is_train,
        )
