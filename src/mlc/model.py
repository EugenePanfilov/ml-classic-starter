from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from mlc.config import ModelConfig, config
from mlc.features import Features


class Model:
    def __init__(
        self,
        features_or_cfg: Optional[Union[dict, Features]] = None,
        model_cfg: Optional[ModelConfig] = None,
    ) -> None:
        if isinstance(features_or_cfg, dict):
            self.features = Features(features_or_cfg)
        elif features_or_cfg is None:
            self.features = Features()
        else:
            self.features = features_or_cfg

        self.cfg = model_cfg or config.model
        self.model = HistGradientBoostingClassifier(**self.cfg.HistGradientBoostingClassifier)

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "Model":
        if self.features.preprocessor is None:
            self.features.fit(X, y)
        X_transformed = self.features.transform(X)
        self.model.fit(X_transformed, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.features.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.features.transform(X))

    def save_model(self, path: Optional[Union[str, Path]] = None) -> None:
        joblib.dump(self.model, Path(path or self.cfg.save_path))

    def save_preprocessor(self, path: Optional[Union[str, Path]] = None) -> None:
        joblib.dump(self.features, Path(path or self.cfg.preprocessor_path))