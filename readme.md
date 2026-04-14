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
│   ├── raw/             # Dados originais (imutáveis)
│   └── processed/       # Dados limpos e transformados
├── docs/                # Documentação do projeto e ML Canvas
├── models/              # Artefatos de modelos (gerados pelo MLflow)
├── notebooks/           # Jupyter Notebooks de EDA e experimentação
├── src/                 # Código-fonte modularizado
│   ├── data/            # Scripts de engenharia de dados
│   └── models/          # Scripts de treino e baselines
├── tests/               # Testes automatizados (Pytest)
├── pyproject.toml       # Gerenciamento de dependências e build
├── setup.ps1            # Automação de ambiente (Windows)
├── Makefile             # Comandos úteis para desenvolvimento
└── README.md            # Este arquivo

---

## 🛠️ Configuração do Ambiente Local

### Prerequisitos
- **Python 3.10+** instalado
- **Windows PowerShell** (para usando setup.ps1) ou terminal Unix-like
- **Git** configurado

### Passo 1: Clonar o Repositório
```powershell
git clone https://github.com/fabiopolli/pos-ml-eng-tech-challenge.git
cd pos-ml-eng-tech-challenge
```

### Passo 2: Setup Automático (Windows)
Execute o script de automação que faz todo o setup de uma vez:

```powershell
.\setup.ps1
```

Este script irá:
1. ✅ Criar um ambiente virtual (`.venv`)
2. ✅ Atualizar o pip
3. ✅ Instalar todas as dependências do projeto

> **Nota:** Se receber um erro de permissão, execute: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Passo 3: Ativar o Ambiente Virtual
Após o setup, ative o ambiente com:

```powershell
.\.venv\Scripts\activate
```

Você saberá que está ativado quando o terminal mostrar `(.venv)` no início da linha.

### Setup Manual (Alternativa - Linux/macOS ou sem PowerShell)
Se preferir fazer manualmente ou está em outro SO:

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar (Linux/macOS)
source .venv/bin/activate

# Ou ativar (Windows - bash)
.\.venv\Scripts\activate

# Instalar dependências
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 🚀 Executando o Pipeline

### Executar o Treino de Baseline
Após ativar o ambiente virtual, execute:

```powershell
python src/models/train_baseline.py
```

Este comando irá:
1. 📊 Carregar o dataset de churn from `data/raw/Telco-Customer-Churn.csv`
2. 🔧 Aplicar pré-processamento (normalização, encoding)
3. 🤖 Treinar dois modelos baseline:
   - **DummyClassifier** (baseline trivial)
   - **LogisticRegression** (modelo inicial)
4. 📈 Executar validação cruzada (5-fold)
5. 💾 Registrar métricas e modelos no MLflow

**Saída esperada:**
```
Carregando os dados...
Configurando o pipeline de pré-processamento...
Iniciando experimentos...

--- Treinando: DummyClassifier_Baseline ---
cv_accuracy: 0.7850
cv_precision: 0.0000
cv_recall: 0.0000
cv_f1_score: 0.0000

--- Treinando: LogisticRegression_Baseline ---
cv_accuracy: 0.8025
cv_precision: 0.6234
cv_recall: 0.5120
cv_f1_score: 0.5621

✅ Concluído! Para ver os resultados, digite 'mlflow ui' no terminal.
```

### Visualizar Resultados no MLflow
Após executar o treino, visualize os experimentos:

```powershell
mlflow ui
```

O MLflow Dashboard abrirá em `http://localhost:5000` onde você pode:
- 📊 Comparar métricas entre runs
- 📈 Analisar parâmetros e performance
- 💾 Baixar modelos treinados

---

## 🧪 Execução de Testes

Execute os testes automatizados com pytest:

```powershell
pytest tests/
```

Para rodas testes com cobertura:

```powershell
pytest tests/ --cov=src
```

---

## 📝 Exploração de Dados (Notebooks)

Execute os notebooks Jupyter para exploração interativa:

```powershell
jupyter notebook
```

Os notebooks disponíveis estão em `notebooks/`:
- **EDA.ipynb** - Análise Exploratória de Dados

---

## 📦 Estrutura de Dependências

As dependências estão organizadas em `pyproject.toml`:

**Dependências de Produção (incluídas no setup):**
- `pandas`, `numpy` - Manipulação de dados
- `scikit-learn` - Algoritmos de ML clássicos
- `torch` - Deep Learning (fase posterior)
- `mlflow` - Rastreamento de experimentos
- `fastapi`, `pydantic`, `uvicorn` - API REST
- E outras (ver `pyproject.toml`)

**Dependências de Desenvolvimento:**
- `pytest` - Testes automatizados
- `ruff` - Linting e formatação
- `ipykernel` - Suporte para Jupyter

---

## 🔗 Workflow Típico de Desenvolvimento

1. **Setup inicial:**
   ```powershell
   .\setup.ps1
   .\.venv\Scripts\activate
   ```

2. **Desenvolver/Corrigir código**

3. **Testar localmente:**
   ```powershell
   pytest tests/
   ```

4. **Rodar experimentos:**
   ```powershell
   python src/models/train_baseline.py
   mlflow ui  # visualizar resultados
   ```

5. **Commit e push:**
   ```powershell
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