import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataProcess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.numerical_cols = [f"f_{i:02d}" for i in range(27)] + ["f_28"]
        self.scaler.fit(df[self.numerical_cols].values)

        # self.fe1_init(df)

    # 14, 19, 23, 25, 28 => (x - mean) ** 2
    # good: 28 > 23 > 25
    def fe1_init(self, df: pd.DataFrame):
        self.fe1_cols = [f"f_{i:02d}" for i in [14, 19, 23, 25, 28]]
        self.fe1_means = {col: df[col].mean() for col in self.fe1_cols}

    def fe1(self, df: pd.DataFrame):
        for col in self.fe1_cols:
            new_col = f"{col}_diff_square"
            df[new_col] = (df[col] / self.fe1_means[col] - 1) ** 2
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols].values)

        # f_29, f_30 -> onehot
        df = df.drop(columns="f_29").join(
            pd.get_dummies(df["f_29"]).rename(columns={0: "f_29_0", 1: "f_29_1"})
        )

        df = df.drop(columns="f_30").join(
            pd.get_dummies(df["f_30"]).rename(
                columns={0: "f_30_0", 1: "f_30_1", 2: "f_30_2"}
            )
        )

        # FE
        # f_14, f_19, f_23, f_25, f_28 -> (x - mean) ** 2
        # df = self.fe1(df)

        # f_27 -> onehot 0, 2, 5 th char
        t1 = pd.get_dummies(df["f_27"].str[0])
        t1.columns = ["f_27_0_A", "f_27_0_B"]

        t2 = pd.get_dummies(df["f_27"].str[2])
        t2.columns = ["f_27_2_A", "f_27_2_B"]

        t3 = pd.get_dummies(df["f_27"].str[5])
        t3.columns = ["f_27_5_A", "f_27_5_B"]

        # df = df.join(t1.join(t2).join(t3))

        # https://www.kaggle.com/code/ambrosm/tpsmay22-gradient-boosting-quickstart/notebook
        # f_27 -> each char to ord
        # f_27 -> nunique
        for i in range(10):
            df[f"f_27_{i}_int"] = df.f_27.str[i].map(ord) - ord("A")
        df[f"f_27_nunique"] = df.f_27.apply(lambda c: len(set(c)))

        df = df.drop(columns="f_27")

        return df

    def postprocess(self, preds: np.ndarray) -> np.ndarray:
        return preds
        # return self.scaler.inverse_transform(preds)
