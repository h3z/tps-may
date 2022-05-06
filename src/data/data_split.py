import pandas as pd
from typing import List
from config.config import RANDOM_STATE
from sklearn.model_selection import train_test_split


def split(df: pd.DataFrame) -> List[pd.DataFrame]:
    train, val = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE)
    return train, val, None
