# 🚀 Rede Neural para Previsão de Churn - Tech Challenge (Fase 1)

Este repositório contém o desenvolvimento de um pipeline profissional de Machine Learning para a previsão de churn em uma empresa de telecomunicações. O projeto faz parte do Tech Challenge da Pós-graduação em Machine Learning Engineering da FIAP.

## 👥 Integrantes do Grupo
* **Fábio Polli** - Infraestrutura & Modelagem Baseline
* **Bill** - Ambiente & Deep Learning (PyTorch)
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
├── notebooks/                # EDA, experimentação e Dashboard
│   ├── EDA.ipynb
│   ├── eda.py
│   └── app_vis.py            # Dashboard interativo (Streamlit)
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

### Pipeline Completo (Quick Start)

Execute o orquestrador principal para treinar todos os modelos em sequência:

```bash
python main.py
```

Isso executa, nessa ordem:
1. **Etapa 1** → `train_baselines.py`: treina `DummyClassifier` e `LogisticRegression`, salvando em `models/`
2. **Etapa 2** → `train_mlp.py`: treina a Rede Neural `MLP` (PyTorch), salvando em `models/`

### Execução Modular (passo a passo)

Caso queira executar etapas individualmente:

```bash
# (Opcional) Processa e exporta CSVs para data/processed/
python src/preprocessing/data_prep.py

# Treina DummyClassifier e LogisticRegression
python src/models/train_baselines.py

# Treina a Rede Neural MLP (PyTorch)
python src/models/train_mlp.py

# Gera métricas finais e gráficos comparativos
python src/models/evaluate_models.py
```

### Visualizar Resultados no MLflow
```bash
mlflow ui
```
O MLflow Dashboard abrirá em `http://localhost:5000` onde você pode comparar métricas entre runs, analisar parâmetros e baixar modelos treinados.

### Dashboard Interativo (Streamlit)
```bash
streamlit run notebooks/app_vis.py
```

---

## 🧪 Execução de Testes

```bash
# Rodar todos os testes
pytest tests/

# Rodar com cobertura de código
pytest tests/ --cov=src
```

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
   pytest tests/
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
- **FIAP Tech Challenge:** [Link do desafio]
- **MLflow Docs:** https://mlflow.org/docs/latest/index.html
- **Scikit-learn:** https://scikit-learn.org/
- **PyTorch:** https://pytorch.org/docs/stable/index.html