import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataProcess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.numerical_cols = [f"f_{i:02d}" for i in range(27)] + ["f_28"]
        self.scaler.fit(df[self.numerical_cols].values)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols].values)

        df = df.drop(columns="f_29").join(
            pd.get_dummies(df["f_29"]).rename(columns={0: "f_29_0", 1: "f_29_1"})
        )

        df = df.drop(columns="f_30").join(
            pd.get_dummies(df["f_30"]).rename(
                columns={0: "f_30_0", 1: "f_30_1", 2: "f_30_2"}
            )
        )

        return df

    def postprocess(self, preds: np.ndarray) -> np.ndarray:
        # return self.scaler.inverse_transform(preds)
        pass
