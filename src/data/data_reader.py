import pandas as pd


class DataReader:
    def __init__(self):
        DATA_ROOT = "dataset"
        self.train = pd.read_pickle(f"{DATA_ROOT}/train.pkl")
        self.test = pd.read_pickle(f"{DATA_ROOT}/test.pkl")
        self.test["target"] = 0
