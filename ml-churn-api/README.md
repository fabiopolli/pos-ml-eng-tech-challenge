# ML Churn API

API para predição de churn de clientes usando FastAPI.

## 📋 Pré-requisitos

- Python 3.10+
- Windows PowerShell ou terminal Linux/Mac

## 🚀 Instalação

### 1. Clonar ou baixar o projeto

```powershell
cd "C:\Users\denis\Desktop\Teste Ambiente Python"
```

### 2. Criar ambiente virtual (venv)

```powershell
python -m venv .venv
```

### 3. Ativar o ambiente virtual

```powershell
# Windows PowerShell
& ".venv\Scripts\Activate.ps1"

# Windows CMD
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### 4. Instalar dependências

```powershell
pip install -r ml-churn-api/requirements.txt
```

Se preferir instalar manualmente:

```powershell
pip install fastapi uvicorn[standard] pydantic pydantic-settings python-dotenv
```

---

## ▶️ Executando a API

### Método 1: Ativar venv primeiro

```powershell
# 1. Ativar ambiente virtual
& ".venv\Scripts\Activate.ps1"

# 2. Entrar na pasta do projeto
cd ml-churn-api

# 3. Rodar o servidor
python -m uvicorn app.main:app --reload --port 8000
```

### Método 2: Executar diretamente com o Python do venv

```powershell
cd ml-churn-api
& ".venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port 8000
```

### Método 3: Com argumentos opcionais

```powershell
cd ml-churn-api
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Acessando a API

Após executar, abra no navegador:

| Serviço | URL |
|---------|-----|
| **Swagger UI** | http://localhost:8000/docs |
| **ReDoc** | http://localhost:8000/redoc |
| **OpenAPI JSON** | http://localhost:8000/openapi.json |

---

## 📡 Endpoints

### Health Check Simples

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" | Select-Object -ExpandProperty Content
```

**Resposta:**
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### Health Check Detalhado

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health/detailed" | Select-Object -ExpandProperty Content
```

### Predição de Churn

```powershell
# PowerShell
$body = @{
    tenure = 12
    monthly_charges = 70.0
    total_charges = 840.0
    contract_type = "one_year"
    payment_method = "credit_card"
    has_phone_service = $true
    has_internet_service = $true
    has_online_security = $false
    has_online_backup = $true
    has_device_protection = $false
    has_tech_support = $false
    streaming_tv = $false
    streaming_movies = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/predict" -Method Post -Body $body -ContentType "application/json"
```

**Resposta:**
```json
{
  "prediction": 0,
  "probability": 0.62,
  "model_version": "1.0.0",
  "request_id": "a1b2c3d4"
}
```

---

## 🧪 Testando com cURL

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Predição
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "monthly_charges": 70.0,
    "total_charges": 840.0,
    "contract_type": "one_year",
    "payment_method": "credit_card",
    "has_phone_service": true,
    "has_internet_service": true,
    "has_online_security": false,
    "has_online_backup": true,
    "has_device_protection": false,
    "has_tech_support": false,
    "streaming_tv": false,
    "streaming_movies": false
  }'
```

---

## 📁 Estrutura do Projeto

```
ml-churn-api/
├── app/
│   ├── main.py                  # FastAPI app
│   ├── core/
│   │   └── config.py            # Configurações
│   ├── schemas/
│   │   ├── predict.py           # Schemas de entrada/saída
│   │   └── health.py            # Schemas de health
│   ├── services/
│   │   └── model_service.py     # Lógica de predição
│   ├── routes/
│   │   ├── predict.py           # Endpoint /predict
│   │   └── health.py            # Endpoint /health
│   └── exceptions/
│       └── handlers.py          # Exception handlers
├── models/                      # Modelos ML (placeholder)
├── tests/                       # Testes
└── requirements.txt             # Dependências
```

---

## ⚙️ Configurações

As configurações estão em [app/core/config.py](ml-churn-api/app/core/config.py):

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `APP_NAME` | ML Churn API | Nome da aplicação |
| `APP_VERSION` | 1.0.0 | Versão |
| `API_PREFIX` | /api/v1 | Prefixo dos endpoints |
| `MODEL_PATH` | models/churn_model.pkl | Caminho do modelo |
| `HOST` | 0.0.0.0 | Host do servidor |
| `PORT` | 8000 | Porta do servidor |

Para customizar, crie um arquivo `.env` na raiz:

```env
APP_NAME=My Churn API
APP_VERSION=2.0.0
PORT=9000
DEBUG=true
```

---

## 🛑 Parar o Servidor

No terminal onde a API está rodando:

```powershell
# Pressione Ctrl+C
```

---

## 🔧 Comandos Úteis

### Verificar se a API está rodando

```powershell
Test-NetConnection -ComputerName localhost -Port 8000
```

### Verificar versão do Python

```powershell
python --version
```

### Listar pacotes instalados

```powershell
pip list
```

### Atualizar dependências

```powershell
pip install --upgrade -r ml-churn-api/requirements.txt
```

---

## 📄 Licença

MIT License