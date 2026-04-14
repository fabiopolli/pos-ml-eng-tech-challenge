Write-Host "⏳ Iniciando o setup do projeto..." -ForegroundColor Cyan

Write-Host "1/3: Criando o ambiente virtual (.venv)..." -ForegroundColor Yellow
python -m venv .venv

Write-Host "2/3: Atualizando o pip na bolha..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "3/3: Instalando as dependências do pyproject.toml..." -ForegroundColor Yellow
.\.venv\Scripts\pip.exe install -e ".[dev]"

Write-Host ""
Write-Host "✅ Setup concluído com sucesso!" -ForegroundColor Green
Write-Host "👉 Para ativar o ambiente agora, copie e cole o comando abaixo:" -ForegroundColor White
Write-Host ".\.venv\Scripts\activate" -ForegroundColor Magenta