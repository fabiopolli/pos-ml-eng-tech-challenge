# Planejamento Estratégico: Tech Challenge Fase 1
**Projeto:** Rede Neural para Previsão de Churn com Pipeline Profissional End-to-End
**Estrutura:** 4 Integrantes | 4 Fases (0 a 3)

## Entregas toda a segunda, reunião para consolidação de informações.

---

### 🗓️ Fase 0: Setup e Alinhamento de Negócio (Semana 0 - 3 dias)
**Dinâmica:** Estabelecer a infraestrutura de desenvolvimento e os requisitos de negócio para assegurar a padronização entre os membros.
**Ferramentas:** GitHub, Python 3.10+, `pyproject.toml`, ML Canvas.

* **Fábio (Infraestrutura):** Criar o repositório institucional no GitHub com a estrutura de diretórios obrigatória: `src/`, `data/`, `models/`, `tests/`, `notebooks/` e `docs/`. CONCLUÍDO
* **Bill (Ambiente e Reprodutibilidade):** Configurar o `pyproject.toml` como única fonte de verdade para dependências e linting. Estabelecer a fixação de *seeds* globais para assegurar a reprodutibilidade dos experimentos. CONCLUÍDO
* **Romário (Definição de Negócio):** Selecionar o dataset (mínimo de 5.000 registros e 10 features) e iniciar o preenchimento do **ML Canvas**, focando em stakeholders e proposta de valor. CONCLUÍDO
* **Denis (Métricas e Governança):** Finalizar o **ML Canvas**, definindo as métricas técnicas (AUC-ROC, F1-Score) e a métrica de negócio (custo de churn evitado). CONCLUÍDO
* **📦 Entrega da Fase 0:** Repositório estruturado, ambiente isolado via `pyproject.toml` e ML Canvas concluído. CONCLUÍDO

---

### 🗓️ Fase 1: EDA e Modelagem Preditiva (Semana 1)
**Dinâmica:** Processamento de dados e desenvolvimento do modelo central e baselines para validação de sinal.
**Ferramentas:** PyTorch, Scikit-Learn, MLflow, Pandas.

* **Romário (Análise Exploratória):** Executar a EDA completa, documentando volume, qualidade, distribuição e prontidão dos dados (*data readiness*). CONCLUÍDO
* **Denis (Engenharia de Dados):** Desenvolver o pipeline de pré-processamento via Scikit-Learn, garantindo a modularização das funções de limpeza e transformação na pasta `src/`. CONCLUÍDO
* **Fábio (Modelagem Baseline):** Treinar modelos baseline (`DummyClassifier` e `LogisticRegression`) utilizando **validação cruzada estratificada**. Registrar os experimentos no MLflow. EM ANDAMENTO
* **Bill (Deep Learning):** Construir e treinar a **Rede Neural (MLP)** com PyTorch, implementando loop de treinamento com *batching* e técnica de *early stopping*. CONCLUÍDO
* **Bill (Documentação):** Atualizar. EM ANDAMENTO
* **Bill (Otimização do modelo):** Atualizar. EM ANDAMENTO
* **📦 Entrega da Fase 1:** Notebook de EDA, scripts de treinamento e tabela comparativa de modelos registrada no MLflow.

---

### 🗓️ Fase 2: Engenharia de Software e API (Semana 2)
**Dinâmica:** Refatoração do código para padrões produtivos e criação da interface de inferência.
**Ferramentas:** FastAPI, Pydantic, Ruff, Pytest, Pandera.

* **Denis (Desenvolvimento de API):** Construir a API utilizando **FastAPI**, implementando as rotas `/predict` e `/health`, além do *middleware* para monitoramento de latência.
* **Bill (Qualidade de Código):** Implementar **logging estruturado** (em substituição ao comando `print()`) e garantir conformidade com o linter **Ruff** (erro zero).
* **Romário (Validação e Automação):** Definir os modelos de dados via **Pydantic** para a API e estruturar o arquivo `Makefile` para automação de tarefas.
* **Fábio (Garantia de Qualidade - QA):** Desenvolver a suíte de testes automatizados com **Pytest**, incluindo *smoke tests*, validação de schema (Pandera) e testes de API.
* **📦 Entrega da Fase 2:** API de inferência funcional, código refatorado em módulos e suíte de testes validada.

---

### 🗓️ Fase 3: Documentação e Finalização (Semana 3)
**Dinâmica:** Consolidação da documentação técnica e defesa do projeto via vídeo explicativo.
**Ferramentas:** Model Card, Método STAR (Vídeo), Markdown.

* **Integrante 1 (Model Card):** Elaborar o **Model Card** completo, detalhando performance, limitações técnicas, vieses e cenários de falha.
* **Integrante 2 (Monitoramento):** Documentar a arquitetura de deploy escolhida (justificando batch vs real-time) e redigir o **plano de monitoramento** (métricas de drift e alertas).
* **Integrante 3 (Documentação Final):** Consolidar o `README.md` com instruções de setup, execução e descrição da arquitetura. (Opcional: Deploy em nuvem).
* **Integrante 4 (Apresentação STAR):** Produzir o vídeo de 5 minutos utilizando o **método STAR** (Situation, Task, Action, Result), demonstrando os resultados e lições aprendidas.
* **📦 Entrega da Fase 3:** Repositório finalizado, documentação técnica completa e link para o vídeo de apresentação.
