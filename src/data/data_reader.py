import pandas as pd


class DataReader:
    def __init__(self):
        DATA_ROOT = "dataset"
        self.train = pd.read_pickle(f"{DATA_ROOT}/train.pkl")
        self.test = pd.read_pickle(f"{DATA_ROOT}/test.pkl")
        self.test["target"] = 0

        self.sample_submission = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")

    def submit(self, preds):
        self.sample_submission.target = preds
        self.sample_submission.to_csv("submit.csv", index=False)
