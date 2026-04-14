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
└── README.md            # Este arquivo