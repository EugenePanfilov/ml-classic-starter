import os
import json

from mlc.config import config
from mlc.data import Data
from mlc.model import Model
from mlc.utils import get_logger, log_metrics, set_deterministic


def main():
    logger = get_logger("train")

    # детерминизм
    set_deterministic(config.data.random_state)

    logger.info("📥 Загружаем данные...")
    data = Data()
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_splits()

    logger.info("⚙️ Строим и обучаем модель...")
    model = Model()
    model.fit(X_train, y_train)

    logger.info("🔎 Оцениваем на валидации...")
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )

    metrics = {
        "roc_auc": roc_auc_score(y_val, y_proba),
        "pr_auc": average_precision_score(y_val, y_proba),
        "accuracy": accuracy_score(y_val, y_pred),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
    }

    # Логируем метрики
    log_metrics(metrics, logger)

    # Сохраняем метрики в artifacts/metrics.json
    os.makedirs("artifacts", exist_ok=True)
    metrics_path = os.path.join("artifacts", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"📊 Метрики сохранены в {metrics_path}")

    logger.info("💾 Сохраняем артефакты...")
    os.makedirs(os.path.dirname(config.model.save_path), exist_ok=True)
    model.save_model(path=config.model.save_path)
    model.save_preprocessor(path=config.model.preprocessor_path)

    logger.info("✅ Обучение завершено")


if __name__ == "__main__":
    main()