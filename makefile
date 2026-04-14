setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install --upgrade pip
	.venv/Scripts/pip install -e ".[dev]"
	@echo "✅ Ambiente virtual criado e dependências instaladas!"