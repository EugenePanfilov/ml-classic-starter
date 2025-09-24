import pandas as pd
import pytest

from mlc.data import Data
from mlc.infer import InferenceModel
from mlc.model import Model


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Обучаем модель и сохраняем артефакты для инференса."""
    Xtr, Xval, Xte, ytr, yval, yte = Data().get_splits()
    model = Model()
    model.fit(Xtr, ytr)

    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    model_path = artifacts_dir / "model.pkl"
    preproc_path = artifacts_dir / "preprocessor.pkl"

    model.save_model(model_path)
    model.save_preprocessor(preproc_path)

    # сохраняем тестовый датасет
    test_path = artifacts_dir / "test.csv"
    pd.concat([Xte, yte.rename("target")], axis=1).to_csv(test_path, index=False)

    return artifacts_dir, test_path


def test_inference_shape(trained_model, tmp_path):
    artifacts_dir, test_path = trained_model
    infer = InferenceModel(artifacts_dir=artifacts_dir)
    preds, proba = infer.predict(
        input_path=str(test_path),
        output_path=str(tmp_path / "preds.csv"),
    )
    df_in = pd.read_csv(test_path)
    assert len(preds) == len(df_in), "⚠️ Число предсказаний не совпадает!"
    assert len(proba) == len(df_in)
