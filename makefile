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

qa-report:
	@echo "Gerando relatorios de QA na pasta tests/docs/..."
	uv run pytest tests/ ml-churn-api/tests/ -v \
		--html=tests/docs/relatorio_qa.html --self-contained-html \
		--cov=src --cov=ml-churn-api/app \
		--cov-report=html:tests/docs/htmlcov > tests/docs/execution_log.txt
	@echo "✅ Relatorios gerados com sucesso na pasta tests/docs/!"