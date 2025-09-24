.PHONY: install lint fmt typecheck test train predict

install:
	poetry install

lint:
	poetry run ruff check .

fmt:
	poetry run ruff format .

typecheck:
	poetry run mypy src/mlc

test:
	poetry run pytest -v

train:
	poetry run python scripts/train.py --config configs/default.yaml

predict:
	poetry run python scripts/predict.py --input artifacts/test.csv --out artifacts/predictions.csv