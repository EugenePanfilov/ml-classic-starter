import os
from pathlib import Path
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd

from mlc.config import config


class InferenceModel:
    def __init__(self, artifacts_dir: Union[str, Path] = "artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.model = joblib.load(self.artifacts_dir / "model.pkl")
        self.preprocessor = joblib.load(self.artifacts_dir / "preprocessor.pkl")

    def load_data(self, input_path: Union[str, Path]) -> pd.DataFrame:
        path = Path(input_path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError("Only .csv or .parquet files are supported")

    def predict(
        self,
        input_path: Union[str, Path, None] = None,
        output_path: Union[str, Path, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        input_path = Path(input_path or config.infer.input_path)
        output_path = Path(output_path or config.infer.output_path)

        df = self.load_data(input_path)
        X_transformed = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X_transformed)[:, 1]
        preds = self.model.predict(X_transformed)

        os.makedirs(output_path.parent, exist_ok=True)
        pd.DataFrame({"proba": proba, "label": preds}).to_csv(output_path, index=False)

        return preds, proba
