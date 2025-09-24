from typing import Optional, Union
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlc.config import FeaturesConfig, config

class Features:
    def __init__(self, cfg: Optional[FeaturesConfig] = None) -> None:
        self.cfg = cfg or config.features
        self.preprocessor: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, None] = None) -> "Features":
        num_cols = self.cfg.num_cols or X.select_dtypes(include="number").columns.tolist()
        cat_cols = self.cfg.cat_cols or []

        transformers = []

        if num_cols:
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(**self.cfg.num_imputer)),
                    ("scaler", StandardScaler(**self.cfg.scaler)),
                ]
            )
            transformers.append(("num", num_pipeline, num_cols))

        if cat_cols:
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(**self.cfg.cat_imputer)),
                    ("encoder", OneHotEncoder(**self.cfg.encoder)),
                ]
            )
            transformers.append(("cat", cat_pipeline, cat_cols))

        self.preprocessor = ColumnTransformer(transformers)
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not fitted yet")
        return self.preprocessor.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, None] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> list[str]:
        if self.preprocessor is None:
            raise ValueError("Preprocessor is not fitted yet")
        return list(self.preprocessor.get_feature_names_out())