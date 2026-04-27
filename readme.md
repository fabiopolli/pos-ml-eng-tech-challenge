# 🚀 Rede Neural para Previsão de Churn - Tech Challenge (Fase 1)

Este repositório contém o desenvolvimento de um pipeline profissional de Machine Learning para a previsão de churn em uma empresa de telecomunicações. O projeto faz parte do Tech Challenge da Pós-graduação em Machine Learning Engineering da FIAP.

## 👥 Integrantes do Grupo
* **Fábio Polli** - Infraestrutura & Modelagem Baseline
* **Bill Kopp** - Ambiente & Deep Learning (PyTorch)
* **Romário** - Definição de Negócio & EDA
* **Denis** - Engenharia de Dados (Pipeline Scikit-Learn)

---

## 📂 Estrutura do Repositório

A organização do projeto segue o padrão *src-layout*, garantindo a separação entre código produtivo, dados e artefatos:

```text
.
├── data/
│   ├── raw/                  # Dados originais (imutáveis)
│   └── processed/            # Dados limpos e transformados
├── docs/                     # Documentação do projeto e ML Canvas
├── models/                   # Modelos persistidos (.pkl, .pth)
├── front/                    # Interface de usuário (Frontend)
│   └── app_vis.py            # Dashboard interativo (Streamlit)
├── notebooks/                # EDA e experimentação inicial
│   └── EDA.ipynb
├── src/                      # Código-fonte modularizado
│   ├── preprocessing/        # Limpeza e engenharia de features
│   │   └── data_prep.py
│   └── models/               # Scripts de treino e avaliação
│       ├── data_utils.py
│       ├── train_baselines.py
│       ├── train_mlp.py
│       └── evaluate_models.py
├── tests/                    # Testes automatizados (Pytest)
├── main.py                   # Orquestrador do pipeline completo
├── pyproject.toml            # Gerenciamento de dependências e build
├── setup.ps1                 # Automação de ambiente (Windows)
├── Makefile                  # Comandos úteis para desenvolvimento
└── README.md                 # Este arquivo
```

---

## 🛠️ Configuração do Ambiente Local

### Pré-requisitos
- **Python 3.10+** instalado
- **Git** configurado

### Passo 1: Clonar o Repositório
```bash
git clone https://github.com/fabiopolli/pos-ml-eng-tech-challenge.git
cd pos-ml-eng-tech-challenge
```

### Passo 2: Criar e Ativar o Ambiente Virtual

**Windows (PowerShell) — setup automático:**
```powershell
.\setup.ps1
.\.venv\Scripts\activate
```
> Se receber erro de permissão: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Linux/macOS — com `uv` (recomendado):**
```bash
uv venv
source .venv/bin/activate
```
> ⚠️ **Atenção:** ao usar `uv venv`, instale com `uv pip` (não `pip`). Misturar os dois causa erro no Debian/Ubuntu (PEP 668).

**Linux/macOS — com `venv` padrão:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Passo 3: Instalar Dependências

**Com `uv` (após `uv venv`):**
```bash
uv pip install -e ".[dev]"
```

**Com `pip` padrão:**
```bash
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 🚀 Executando o Pipeline

Execute o orquestrador principal para treinar todos os modelos em sequência. Você pode fazer isso de duas formas:

#### Opção A: Via Python (Direto)
```bash
python main.py
```

#### Opção B: Via MLflow Projects (Recomendado)
O projeto está configurado com um arquivo `MLproject`, permitindo a execução padronizada:
```bash
mlflow run . --env-manager=local
```

Isso executa, nessa ordem:
1. **Validação de Dados** → Verifica o esquema do CSV com Pandera.
2. **Etapa 1** → `train_baselines.py`: treina `DummyClassifier` e `LogisticRegression`.
3. **Etapa 2** → `train_mlp.py`: treina a Rede Neural `MLP` (PyTorch).

> 💡 **Nota:** Todos os artefatos (modelos e scalers) são salvos na pasta `models/` e registrados automaticamente no **MLflow Tracking**.

### Execução Modular (passo a passo)

Caso queira executar etapas individualmente:

```bash
# Valida os dados brutos com Pandera e exporta CSVs para data/processed/
python src/preprocessing/data_prep.py

# Treina DummyClassifier e LogisticRegression
python src/models/train_baselines.py

# Treina a Rede Neural MLP (PyTorch)
python src/models/train_mlp.py

# Gera métricas finais e gráficos comparativos
python src/models/evaluate_models.py
```

### 📊 Gerenciamento de Experimentos (MLflow)

O projeto utiliza o **MLflow** para gerenciar todo o ciclo de vida do modelo. Ao executar o pipeline (via `main.py` ou `mlflow run`), o MLflow registra automaticamente:

*   **Parâmetros:** Hiperparâmetros, sementes de aleatoriedade e configurações.
*   **Métricas:** Acurácia, perda (loss) por época e métricas de validação cruzada.
*   **Modelos:** Registro automático no **Model Registry** para versionamento.
*   **Nested Runs:** O pipeline principal cria uma "Run Pai" que engloba as sub-etapas (Baselines, MLP e Avaliação) para melhor organização.

#### Como visualizar os resultados:
1. Inicie a interface visual:
   ```bash
   make mlflow-ui   # ou: mlflow ui
   ```
2. Acesse no navegador: **[http://localhost:5000](http://localhost:5000)**

#### O que observar no Dashboard:
*   **Curvas de Aprendizado:** Na run `mlp_run`, analise o gráfico de `train_loss` vs `val_loss` para detectar overfitting.
*   **Comparação:** Selecione múltiplas runs e use o botão **Compare** para ver o impacto de diferentes hiperparâmetros.
*   **Model Registry:** Acesse a aba **Models** para gerenciar versões dos modelos `ChurnMLP` e `ChurnLogisticRegression`.

#### Avaliação Final:
Para consolidar as métricas de teste e gerar o gráfico comparativo final no MLflow:
```bash
make eval  # ou: python src/models/evaluate_models.py
```
Isso criará uma `evaluation_run` com o relatório `evaluation_summary.png` salvo como um artefato.

> 📚 **Dica:** Para um guia passo a passo detalhado, consulte o [Tutorial MLflow](docs/tutorial_mlflow.md).

### API de Predição (FastAPI)
```bash
# Sincronizar ambiente (caso necessário)
uv sync --all-extras

# Rodar o servidor da API
uv run uvicorn app.main:app --app-dir ml-churn-api --reload
```
A API estará disponível em `http://localhost:8000`. 
- **Documentação Interativa (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Documentação Alternativa (Redoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔗 Detalhamento da API (Endpoints e Contratos)

A API foi construída com **FastAPI** e segue padrões RESTful, com validação rigorosa de tipos via **Pydantic**.

### 1. Principais Endpoints
- `GET /api/v1/health`: Verifica se o serviço está online.
- `POST /api/v1/predict`: Recebe dados do cliente e retorna a probabilidade de churn.

### 2. Dicionário de Dados (Entrada - `/predict`)

O corpo da requisição (`JSON body`) deve conter os seguintes campos:

| Campo | Tipo | Descrição | Intervalo / Opções |
| :--- | :--- | :--- | :--- |
| `tenure` | `int` | Meses de permanência do cliente | `0` a `120` |
| `monthly_charges` | `float` | Valor da mensalidade atual | `0` a `500.0` |
| `total_charges` | `float` | Gasto total acumulado | `0` a `50000.0` |
| `contract_type` | `str` | Tipo de contrato do cliente | `monthly`, `one_year`, `two_year` |
| `payment_method` | `str` | Método de pagamento | `credit_card`, `debit_card`, `electronic_check`, `bank_transfer` |
| `has_phone_service` | `bool` | Se possui serviço de telefone | `true` ou `false` |
| `has_internet_service`| `bool` | Se possui serviço de internet | `true` ou `false` |
| `has_online_security` | `bool` | Se possui segurança online | `true` ou `false` |
| `has_tech_support` | `bool` | Se possui suporte técnico | `true` ou `false` |
| `streaming_tv` | `bool` | Se assina streaming de TV | `true` ou `false` |
| `streaming_movies` | `bool` | Se assina streaming de filmes | `true` ou `false` |

### 3. Exemplo de Uso (Requisição e Resposta)

**Requisição (cURL):**
```bash
curl -X 'POST' 'http://localhost:8000/api/v1/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "tenure": 24,
    "monthly_charges": 85.0,
    "total_charges": 2040.0,
    "contract_type": "one_year",
    "payment_method": "credit_card",
    "has_online_security": true
  }'
```

**Resposta (Sucesso - 200 OK):**
```json
{
  "prediction": 1,
  "probability": 0.7452,
  "model_version": "1.0.0",
  "request_id": "a1b2c3d4"
}
```

### 4. Usando o Swagger UI (Interativo)
Ao acessar `/docs`, você verá a interface do Swagger. Para testar:
1. Clique no botão **"Try it out"** no endpoint `/predict`.
2. Edite o JSON de exemplo com os valores desejados.
3. Clique em **"Execute"**.
4. Veja o resultado (Response Body) e o código de status logo abaixo.

---

### Dashboard Interativo (Streamlit)
```bash
uv run streamlit run front/app_vis.py
```
O dashboard agora conta com uma aba exclusiva de **Análise de Custo-Benefício**, permitindo simular o ROI do modelo com base em parâmetros de negócio (Ticket Médio e Custo de Retenção).

---

## 🧪 Execução de Testes (QA & Validação)

A suíte de testes automatizados foi desenvolvida focando na garantia de qualidade (QA) contínua do pipeline de dados e da API, utilizando **Pytest**. A robustez do sistema é garantida por:

*   **Smoke Tests:** Validação ponta a ponta do treinamento dos modelos (Baselines e MLP PyTorch), garantindo que os artefatos sejam gerados corretamente.
*   **Validação de Dados (Fail Fast):** Schemas do **Pandera** garantem a integridade dos dados brutos antes do processamento.
*   **Contratos de API:** Validação rigorosa de endpoints e payloads utilizando **Pydantic**.
*   **Compatibilidade:** Fixtures que garantem a execução dos testes tanto em Windows quanto em sistemas Unix (Linux/macOS).

**Execução Rápida (Local):**
```bash
pytest tests/ ml-churn-api/tests/ -v
```

**Gerar Relatórios Visuais (HTML & Coverage):**
```bash
# Para utilizadores Linux/macOS:
make qa-report

# Para utilizadores Windows (PowerShell):
New-Item -ItemType Directory -Force -Path tests\docs; pytest tests/ ml-churn-api/tests/ -v --html=tests/docs/relatorio_qa.html --self-contained-html --cov=src --cov=ml-churn-api/app --cov-report=html:tests/docs/htmlcov
```
Os relatórios interativos serão guardados na pasta `tests/docs/`.

### 🤖 Integração Contínua (CI/CD - GitHub Actions)
O projeto possui uma esteira configurada no **GitHub Actions** (`.github/workflows/ci_pipeline.yml`). 
A cada `push` ou `pull request` na branch `main`:
1. Um ambiente isolado é criado na nuvem.
2. As dependências são instaladas via `uv`.
3. Todos os testes são executados automaticamente utilizando **Mocks dinâmicos** para simular os modelos da API, garantindo a validação da lógica sem trafegar arquivos pesados.
4. O *pipeline* bloqueia merges caso a cobertura de testes ou algum schema falhe.

---

## 📝 Exploração de Dados (Notebooks)

```bash
jupyter notebook
```

Os notebooks disponíveis estão em `notebooks/`:
- **EDA.ipynb** — Análise Exploratória de Dados interativa

---

## 📦 Estrutura de Dependências

As dependências estão organizadas em `pyproject.toml`:

**Produção:**
- `pandas`, `numpy` — Manipulação de dados
- `scikit-learn` — Algoritmos de ML clássicos
- `torch` — Deep Learning (MLP)
- `joblib` — Serialização de modelos
- `mlflow` — Rastreamento de experimentos
- `pandera` — Validação de schemas de dados
- `streamlit` — Dashboard interativo
- `fastapi`, `pydantic`, `uvicorn` — API REST
- `matplotlib`, `seaborn` — Visualização

**Desenvolvimento:**
- `pytest` — Testes automatizados
- `ruff` — Linting e formatação
- `ipykernel` — Suporte para Jupyter

---

## 🔗 Workflow Típico de Desenvolvimento

1. **Setup inicial:**
   ```bash
   # Linux
   uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
   # Windows
   .\setup.ps1 && .\.venv\Scripts\activate
   ```

2. **Desenvolver/Corrigir código**

3. **Testar localmente:**
   ```bash
pytest tests/ ml-churn-api/tests/ -v
   ```

4. **Rodar o pipeline completo:**
   ```bash
   python main.py
   python src/models/evaluate_models.py
   mlflow ui  # visualizar resultados
   ```

5. **Commit e push:**
   ```bash
   git add .
   git commit -m "Descrição das mudanças"
   git push origin main
   ```

---

## 📚 Recursos Adicionais

- **ML Canvas:** Documentação de negócio em `docs/`
- **Relatório de Implementação:** Detalhes técnicos em `docs/relatorio_de_implementacao.md`
- **Guia da API FastAPI:** Manual de uso da API em `docs/guia_api_fastapi.md`
- **Tutorial MLflow:** Guia rápido para o time em `docs/tutorial_mlflow.md`
- **FIAP Tech Challenge:** [Link do desafio]
- **MLflow Docs:** https://mlflow.org/docs/latest/index.html
- **Scikit-learn:** https://scikit-learn.org/
- **PyTorch:** https://pytorch.org/docs/stable/index.html