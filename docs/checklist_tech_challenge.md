# Checklist do Projeto: Tech Challenge - Previsão de Churn

Este checklist serve como guia para garantir que todos os requisitos técnicos e de negócio da Fase 01 da pós-graduação sejam atendidos.

## 📂 1. Estrutura e Governança do Código
- [x] Criar estrutura de pastas padrão: `src/`, `data/`, `models/`, `tests/`, `notebooks/`, `docs/`.
- [x] Configurar `pyproject.toml` como única fonte de verdade para dependências.
- [x] Configurar `.gitignore` (ignorar arquivos `.env`, pastas de dados pesados e `__pycache__`).
- [x] Garantir histórico de commits limpo (mensagens claras e commits frequentes).
- [x] Configurar Linting e Formatação (Ruff).
- [x] Implementar Logging estruturado (substituindo todos os `print()`).
- [x] Fixar sementes (seeds) para reprodutibilidade total.
- [x] Criar um `Makefile` com comandos (`make lint`, `make test`, `make run`).

## 📊 2. Etapa 1: Entendimento e Preparação (EDA & Baseline)
- [ ] Preencher ML Canvas (Stakeholders, Métricas, SLOs).
- [x] Notebook de EDA: Análise de volume, qualidade e correlação das features.
- [x] Definir métrica técnica (ex: AUC-ROC) e métrica de negócio (ex: economia de churn).
- [x] Implementar Baselines com Scikit-Learn (Dummy e Regressão Logística).
- [x] Configurar rastreamento inicial no MLflow.

## 🧠 3. Etapa 2: Modelagem com Redes Neurais (PyTorch)
- [x] Implementar arquitetura MLP (Multi-Layer Perceptron) no PyTorch.
- [x] Implementar Loop de Treinamento com Batching e Early Stopping.
- [x] Realizar Validação Cruzada Estratificada.
- [x] Comparar MLP vs Baselines (mínimo 4 métricas).
- [x] Realizar análise de custo/benefício (Trade-off de erros).
- [x] Registrar todos os experimentos da MLP no MLflow.

## ⚙️ 4. Etapa 3: Engenharia e API (FastAPI)
- [x] Refatorar código dos notebooks para módulos na pasta `src/`.
- [x] Criar Pipeline de pré-processamento robusto (Sklearn Pipeline).
- [PARCIAL] Escrever Testes Automatizados com Pytest (unitários e smoke tests).
- [PARCIAL] Implementar validação de dados de entrada com Pandera.
- [x] Desenvolver API FastAPI com endpoints `/predict` e `/health`.
- [PARCIAL] Implementar validação de schemas de entrada/saída com Pydantic.
- [x] Adicionar Middleware de monitoramento de latência na API.

## 📝 5. Etapa 4: Documentação e Entrega
- [ ] Escrever Model Card (Performance, limitações e vieses).
- [ ] Elaborar Plano de Monitoramento (Métricas, alertas e playbook).
- [PARCIAL] Finalizar README.md (Instruções de setup e uso).
- [ ] Gravar Vídeo STAR (máximo 5 minutos).
- [ ] (Bônus) Deploy da API em nuvem (AWS/Azure/GCP) com URL pública.

---
*Gerado pelo seu assistente Python, Data.*
