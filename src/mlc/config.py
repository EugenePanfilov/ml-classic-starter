from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    test_path: Path = Path("artifacts/test.csv")


@dataclass
class FeaturesConfig:
    num_cols: Optional[list[str]] = None
    cat_cols: Optional[list[str]] = None
    num_imputer: Dict[str, Any] = field(default_factory=lambda: {"strategy": "median"})
    cat_imputer: Dict[str, Any] = field(
        default_factory=lambda: {"strategy": "most_frequent"}
    )
    scaler: Dict[str, Any] = field(default_factory=dict)
    encoder: Dict[str, Any] = field(
        default_factory=lambda: {"handle_unknown": "ignore"}
    )


@dataclass
class ModelConfig:
    HistGradientBoostingClassifier: Dict[str, Any] = field(
        default_factory=lambda: {"random_state": 42}
    )
    save_path: Path = Path("artifacts/model.pkl")
    preprocessor_path: Path = Path("artifacts/preprocessor.pkl")


@dataclass
class InferConfig:
    artifacts_dir: Path = Path("artifacts")
    input_path: Path = Path("artifacts/test.csv")
    output_path: Path = Path("artifacts/preds.csv")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data": vars(self.data),
            "features": vars(self.features),
            "model": vars(self.model),
            "infer": vars(self.infer),
        }


def load_config(path: str = "configs/default.yaml") -> Config:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")
    with open(path_obj, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    return Config(
        data=DataConfig(**raw_cfg.get("data", {})),
        features=FeaturesConfig(**raw_cfg.get("features", {})),
        model=ModelConfig(**raw_cfg.get("model", {})),
        infer=InferConfig(**raw_cfg.get("infer", {})),
    )


config = load_config()