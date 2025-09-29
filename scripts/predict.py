# scripts/predict.py
from mlc.config import config
from mlc.infer import InferenceModel
from mlc.utils import get_logger


def main():
    logger = get_logger("predict")

    logger.info("📥 Загружаем модель и делаем предсказания...")
    infer = InferenceModel()

    preds, proba = infer.predict(
        input_path=config.infer.input_path,
        output_path=config.infer.output_path,
    )

    logger.info(f"✅ Предсказания сохранены в {config.infer.output_path}")
    logger.info(f"🔢 Кол-во предсказаний: {len(preds)}")


if __name__ == "__main__":
    main()