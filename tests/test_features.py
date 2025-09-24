import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from mlc.features import Features


@pytest.fixture(scope="module")
def data():
    ds = load_breast_cancer(as_frame=True)
    X, y = ds.data, ds.target
    return X, y


def test_no_target_leakage(data):
    X, y = data
    features = Features()
    features.fit(X, y)
    names = features.get_feature_names_out()
    assert all("target" not in n for n in names), "⚠️ Таргет попал в препроцессинг!"


def test_shape_and_type(data):
    X, y = data
    features = Features()
    Xt = features.fit_transform(X, y)
    assert Xt.shape[0] == X.shape[0]
    assert isinstance(Xt, (np.ndarray, np.generic)), "Выход не numpy!"
