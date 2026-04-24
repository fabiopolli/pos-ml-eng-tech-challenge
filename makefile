.PHONY: setup run eval test test-full mlflow-ui

setup:
	uv venv
	uv pip install --upgrade pip
	uv pip install -e ".[dev]"
	@echo "✅ Ambiente virtual criado e dependências instaladas com uv!"

run:
	uv run python main.py

eval:
	uv run python src/models/evaluate_models.py

test:
	uv run python -m pytest tests/ -v -m "not slow"

test-full:
	uv run python -m pytest tests/ -v

mlflow-ui:
	@echo "Iniciando MLflow UI em http://localhost:5000 ..."
	uv run mlflow ui