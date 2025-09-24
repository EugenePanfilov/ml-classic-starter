## 🚀 Быстрый старт (3 шага)

```bash
# 1. Установить зависимости
poetry install

# 2. Обучить модель
poetry run python scripts/train.py

# 3. Сделать предсказания
poetry run python scripts/predict.py


# Описание модулей

configs/ – хранит YAML-конфиги с гиперпараметрами.

scripts/train.py – запускает полный цикл обучения модели.

scripts/predict.py – делает инференс, генерирует predictions.csv.

mlc/config.py – загружает конфиг.

mlc/data.py – загружает и разбивает данные.

mlc/features.py – препроцессинг (импьютер, скейлинг, OHE).

mlc/model.py – строит и обучает модель.

mlc/infer.py – загружает модель и делает предсказания.

mlc/utils.py – логирование и фиксация random seed.

tests/ – модульные и интеграционные тесты PyTest.


# Тестирование

Запустить все тесты:

poetry run pytest -v