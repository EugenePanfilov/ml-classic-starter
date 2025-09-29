![CI](https://github.com/EugenePanfilov/ml-classic-starter/actions/workflows/ci.yml/badge.svg)


# 🚀 Быстрый старт (3 шага)

### 1. Установить зависимости
```bash
make install
```
### 2. Обучить модель
```bash
make train
```
### 3. Сделать предсказания
```bash
make predict
```


# Основные команды (Makefile)
```bash
make install   # Установка зависимостей
make lint      # Проверка стиля (ruff)
make fmt       # Автоформатирование (ruff)
make typecheck # Проверка типов (mypy)
make test      # Запуск тестов (pytest + coverage)
make train     # Обучение модели
make predict   # Инференс
```


# Модули

config.py — загрузка и хранение конфигов (YAML → dataclass).

data.py — загрузка датасета breast_cancer, генерация train/val/test, сохранение test.csv.

features.py — препроцессинг (числовые + категориальные признаки, scaler, OHE).

model.py — обёртка над GradientBoostingClassifier + сохранение артефактов.

infer.py — загрузка модели и предсказание на новых данных.

utils.py — логирование и установка детерминизма (фиксация numpy, random)


# Тесты

Запуск тестов:
```bash
make test
```

Покрывают:

- отсутствие утечки таргета в препроцессинге,

- метрики модели (ROC-AUC ≥ 0.80 на hold-out),

- детерминизм при одинаковом random_state,

- корректность инференса.


## ⚙️ CI/CD

В проект встроен минимальный **GitHub Actions pipeline**, который проверяет качество кода и тесты при каждом пуше или pull request.

Workflow выполняет следующие шаги:

1. **Установка зависимостей** (poetry install)
2. **Линтинг** (ruff check .)
3. **Проверка типов** (mypy src)
4. **Тесты** (pytest -q --cov=src/mlc)

### 🔧 Пример workflow

Файл: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install Poetry
        run: pip install poetry

      - name: Install dependencies
        run: poetry install --with dev

      - name: Lint
        run: poetry run ruff check .

      - name: Typecheck
        run: poetry run mypy src

      - name: Run tests
        run: poetry run pytest -q --cov=src/mlc


# Пример логов

```bash
$ make train
✅ Test saved to artifacts/test.csv
✅ Model saved to artifacts/model.pkl
✅ Preprocessor saved to artifacts/preprocessor.pkl

$ make predict
✅ Predictions saved to artifacts/predictions.csv
INFO - Количество предсказаний: 114
```