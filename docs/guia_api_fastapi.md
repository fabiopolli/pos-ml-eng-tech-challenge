# 🌐 Guia de Implementação: API de Predição (FastAPI)

Este guia descreve a arquitetura e o funcionamento da API desenvolvida para servir os modelos de predição de churn. A API foi projetada seguindo padrões de mercado, garantindo performance, segurança e consistência com o pipeline de treinamento.

---

## 1. Arquitetura do Projeto

A API está localizada em `ml-churn-api/` e utiliza uma estrutura modular:

```text
ml-churn-api/
├── app/
│   ├── core/           # Configurações globais (pydantic-settings)
│   ├── exceptions/     # Tratamento centralizado de erros
│   ├── middleware/     # Segurança, Logs e Rate Limiting
│   ├── models/         # Carregamento de modelos e Preprocessamento
│   ├── routes/         # Definição dos endpoints (Health, Predict)
│   ├── schemas/        # Contratos de dados (Pydantic)
│   ├── services/       # Lógica de negócio e orquestração
│   └── main.py         # Ponto de entrada da aplicação
```

---

## 2. O Fluxo de uma Predição (Passo a Passo)

Quando um cliente envia um JSON para o endpoint `/predict`, o seguinte fluxo ocorre:

1.  **Validação (Schemas):** O FastAPI utiliza o `PredictInput` (Pydantic) para validar se todos os campos estão presentes. Implementamos limites flexíveis para evitar erros com novos dados (ex: `tenure` até 120 meses, `monthly_charges` até $500). Além disso, usamos `Literal` para garantir que os tipos de contrato e pagamento sejam exatamente os esperados pela API.
2.  **Orquestração (Service):** O `model_service.py` recebe os dados validados e gera um `request_id` único para rastreamento.
3.  **Preprocessamento de Inferência:** 
    - O `inference_preprocessor.py` entra em ação.
    - Ele converte o input da API no formato do dataset original.
    - Aplica as mesmas funções de **Feature Engineering** do pipeline de treino.
    - Aplica o **One-Hot Encoding** alinhado com as colunas do treinamento.
    - Utiliza o `scaler.pkl` para normalizar os valores numéricos.
4.  **Carregamento Inteligente do Modelo:**
    - A API tenta carregar o modelo `ChurnLogisticRegression` do **MLflow Model Registry**.
    - Se falhar, ela busca automaticamente o arquivo `models/logistic_model.pkl` local como fallback.
5.  **Inferência:** O modelo processa as features e retorna a predição (0 ou 1) e a probabilidade.
6.  **Resposta:** A API retorna um objeto `PredictOutput` estruturado.

---

## 3. Funcionalidades de Produção

### 🛡️ Middlewares
- **LoggingMiddleware:** Registra o tempo de resposta de cada requisição e o status HTTP.
- **SecurityHeadersMiddleware:** Adiciona cabeçalhos de segurança (XSS Protection, Content Type Options).
- **RateLimitMiddleware:** (Opcional) Protege a API contra excesso de requisições.

### 🚑 Tratamento de Erros
As exceções são capturadas por handlers globais em `app/exceptions/handlers.py`, garantindo que o cliente receba sempre um JSON padronizado com o erro, em vez de um "Internal Server Error" genérico ou uma stack trace exposta.

### 🔍 Monitoramento (Health Check)
- `/`: Redireciona automaticamente para a documentação interativa (/docs).
- `/api/v1/health`: Retorna se a API está de pé.
- `/api/v1/health/detailed`: Verifica se o modelo foi carregado corretamente e está pronto para inferência.

---

## 4. Como Utilizar

### Executando o Servidor
Certifique-se de estar na raiz do projeto e use o `uv`:

```bash
uv run uvicorn app.main:app --app-dir ml-churn-api --reload
```

### Documentação Interativa
O FastAPI gera automaticamente documentações de alta qualidade:
- **Swagger UI:** `http://localhost:8000/docs` (Permite testar a API direto pelo navegador).
- **ReDoc:** `http://localhost:8000/redoc`

### Exemplo de Requisição (cURL)
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "tenure": 24,
  "monthly_charges": 85.5,
  "total_charges": 2050.0,
  "contract_type": "one_year",
  "payment_method": "credit_card",
  "has_phone_service": true,
  "has_internet_service": true,
  "has_online_security": true,
  "has_online_backup": false,
  "has_device_protection": true,
  "has_tech_support": true,
  "streaming_tv": true,
  "streaming_movies": false
}'
```

---

## 5. Próximos Passos Sugeridos

1.  **Monitoramento de Drift:** Implementar o `evidently` (já presente nas dependências) para monitorar se os dados que chegam na API estão mudando muito em relação aos dados de treino.
2.  **Autenticação:** Adicionar uma camada de API Key para proteger os endpoints de predição.
3.  **Dockerização:** Criar um `Dockerfile` para facilitar o deploy da API em ambientes de nuvem.
