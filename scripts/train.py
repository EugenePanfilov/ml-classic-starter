import os

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
    proba_val = model.predict_proba(X_val)[:, 1]
    preds_val = model.predict(X_val)

    from sklearn.metrics import accuracy_score, roc_auc_score

    metrics = {
        "roc_auc_val": float(roc_auc_score(y_val, proba_val)),
        "accuracy_val": float(accuracy_score(y_val, preds_val)),
    }

    # Логируем метрики
    log_metrics(metrics, logger)

    logger.info("💾 Сохраняем артефакты...")
    os.makedirs(os.path.dirname(config.model.save_path), exist_ok=True)
    model.save_model(path=config.model.save_path)
    model.save_preprocessor(path=config.model.preprocessor_path)

    logger.info("✅ Обучение завершено")


if __name__ == "__main__":
    main()
