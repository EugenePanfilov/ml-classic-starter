from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit

from mlc.config import config


class Data:
    def __init__(self, cfg=None) -> None:
        cfg = cfg or config.data
        self.test_size = cfg.test_size
        self.val_size = cfg.val_size
        self.random_state = cfg.random_state
        self.test_path = Path(cfg.test_path)

        np.random.seed(self.random_state)
        self.data = load_breast_cancer(as_frame=True)
        self.X = self.data.data
        self.y = self.data.target
        self.splits: (
            Tuple[
                pd.DataFrame,
                pd.DataFrame,
                pd.DataFrame,
                pd.Series,
                pd.Series,
                pd.Series,
            ]
            | None
        ) = None

    def get_splits(
        self,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        if self.splits is not None:
            return self.splits

        splitter_test = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=self.random_state
        )
        train_val_idx, test_idx = next(splitter_test.split(self.X, self.y))
        X_train_val, X_test = self.X.iloc[train_val_idx], self.X.iloc[test_idx]
        y_train_val, y_test = self.y.iloc[train_val_idx], self.y.iloc[test_idx]

        val_relative_size = self.val_size / (1 - self.test_size)
        splitter_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_relative_size, random_state=self.random_state
        )
        train_idx, val_idx = next(splitter_val.split(X_train_val, y_train_val))
        X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

        self.splits = (X_train, X_val, X_test, y_train, y_val, y_test)

        pd.concat([X_test, y_test.rename("target")], axis=1).to_csv(
            self.test_path, index=False
        )
        print(f"Test saved to {self.test_path}")

        return self.splits
