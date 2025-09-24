import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from mlc.model import Model

RND = 42


@pytest.fixture(scope="module")
def data():
    ds = load_breast_cancer(as_frame=True)
    return ds.data, ds.target


@pytest.fixture(scope="module")
def splits(data):
    X, y = data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=RND)
    train_idx, test_idx = next(sss.split(X, y))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def test_fit_and_auc(splits):
    Xtr, Xte, ytr, yte = splits
    model = Model()
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    assert auc > 0.8, f"ROC-AUC слишком низкий: {auc:.3f}"


def test_determinism(splits):
    Xtr, Xte, ytr, yte = splits
    aucs = []
    for seed in [RND, RND]:
        model = Model()
        model.model.set_params(random_state=seed)
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, proba))
    assert np.allclose(aucs[0], aucs[1], atol=1e-6), "Результаты не детерминированы!"
