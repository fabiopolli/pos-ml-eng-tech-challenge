# 📡 Plano de Monitoramento — Modelo de Previsão de Churn

## Visão Geral

Um modelo de Machine Learning em produção **não é um entregável estático**. Sem monitoramento contínuo, o modelo degrada silenciosamente: os perfis dos clientes mudam, os planos da operadora são atualizados e a distribuição dos dados de entrada se distancia daquela com a qual o modelo foi treinado. O resultado direto é um aumento de **Falsos Negativos** — clientes que cancelam sem que o modelo os identifique — gerando prejuízo financeiro real.

Este plano define três pilares de observabilidade:
1. **Métricas** — os indicadores que vamos acompanhar.
2. **Alertas** — os gatilhos automáticos que disparam quando algo sai do esperado.
3. **Playbook** — o manual de ação para cada tipo de incidente.

---

## 1. Métricas

### 1.1 Métricas de Sistema (API FastAPI)

Métricas coletadas pela camada de middleware já implementada na API (`LoggingMiddleware` e `X-Process-Time`).

| Métrica | O que mede | Como é coletada hoje | Meta (SLO) |
| :--- | :--- | :--- | :--- |
| **Latência (P95)** | Tempo de resposta do endpoint `/api/v1/predict` | Header `X-Process-Time` emitido pelo `LoggingMiddleware` a cada requisição | ≤ 200ms |
| **Taxa de Erros** | Proporção de respostas HTTP 5xx sobre o total | Logs estruturados do `LoggingMiddleware` (campo `status=`) | < 1% |
| **Throughput** | Volume de requisições por segundo (RPS) | Contagem de logs `Requisição recebida` por intervalo de tempo | Baseline a definir |
| **Rate Limit** | Requisições rejeitadas (HTTP 429) por abuso de uso | `RateLimitMiddleware` (headers `X-RateLimit-Remaining`) | < 5% do total |
| **Disponibilidade** | API respondendo no endpoint `/api/v1/health` | Health Check simples + detalhado (`/health/detailed`) já implementados | ≥ 99.5% |

### 1.2 Métricas de Dados (*Data Drift*)

Monitoram se os dados que chegam à API em produção continuam com o mesmo perfil estatístico dos dados de treinamento.

| Métrica | O que mede | Features críticas monitoradas | Detecção |
| :--- | :--- | :--- | :--- |
| **Drift de Distribuição** | Deslocamento estatístico nas variáveis de entrada em relação à distribuição do treino | `tenure`, `MonthlyCharges`, `TotalCharges` (features numéricas normalizadas pelo `StandardScaler`) | Teste KS (Kolmogorov-Smirnov) ou PSI (Population Stability Index) em janela semanal |
| **Drift em Features Engenheiradas** | Mudança no comportamento das features criadas no `apply_feature_engineering()` | `Services_Count`, `Charge_Difference`, `Is_Electronic_Check` | Comparação de média/desvio contra baseline do treino |
| **Drift Categórico** | Surgimento de categorias novas ou desaparecimento de categorias esperadas | `Contract`, `InternetService`, `PaymentMethod` (validados pelo Pandera `RawDataSchema`) | Verificação contra o `isin` definido no schema |
| **Valores Inválidos** | Dados fora do contrato de schema na entrada da API | Todos os campos do `PredictInput` (Pydantic) | Validação automática pelo Pydantic (retorna HTTP 422); monitorar volume de rejeições |

### 1.3 Métricas do Modelo (*Model Drift*)

Monitoram a qualidade das predições ao longo do tempo.

| Métrica | O que mede | Baseline de Produção | Fonte |
| :--- | :--- | :--- | :--- |
| **F1-Score (weighted)** | Equilíbrio entre Precisão e Recall no conjunto mais recente com *ground truth* | Valor logado no MLflow como `lr_test_f1` ou `mlp_test_f1` na run `evaluation_run` | Avaliação semanal contra dados confirmados |
| **AUC-ROC** | Capacidade de separação entre classes em diferentes limiares | Valor logado como `lr_test_auc_roc` / `mlp_test_auc_roc` | Avaliação semanal |
| **Recall (weighted)** | Proporção de churners reais que o modelo detectou — métrica mais importante para retenção | Valor logado como `lr_test_recall` / `mlp_test_recall` | Avaliação semanal |
| **Distribuição de Predições** | % de predições positivas (Churn=1) sobre o volume total | Baseline histórico (~27%, refletindo o desbalanceamento do dataset de treino) | Agregação diária sobre respostas do `/predict` |
| **Economia Estimada (Savings)** | Impacto financeiro real do modelo, baseado na Matriz de Confusão | Valor logado como `lr_test_estimated_savings` (premissas: Ticket=$70, Retenção=$30) | Recalculado semanalmente com *ground truth* atualizado |

---

## 2. Alertas

### 2.1 Regras de Disparo

| Severidade | Alerta | Gatilho (Threshold) | Canal de Notificação |
| :---: | :--- | :--- | :--- |
| 🔴 **Crítico** | API Indisponível | Endpoint `/api/v1/health` retorna falha ou timeout por > 2 minutos | Slack #incidents + E-mail do Squad |
| 🔴 **Crítico** | Modelo Não Carregado | `/api/v1/health/detailed` retorna `model: null` (falha no MLflow Registry **e** no fallback local) | Slack #incidents |
| 🟠 **Alto** | Latência Degradada | P95 do `/predict` > 500ms por mais de 5 minutos contínuos | Slack #alerts |
| 🟠 **Alto** | Taxa de Erros Elevada | Erros HTTP 5xx > 5% em janela de 10 minutos | Slack #alerts |
| 🟡 **Médio** | Data Drift Detectado | Teste estatístico (PSI > 0.2) em qualquer das 3 features numéricas críticas (`tenure`, `MonthlyCharges`, `TotalCharges`) | Slack #ml-monitoring |
| 🟡 **Médio** | Queda de Performance | F1-Score cai > 5% em relação ao baseline da `evaluation_run` do MLflow | Slack #ml-monitoring |
| 🟡 **Médio** | Anomalia na Distribuição de Predições | Proporção de Churn=1 desvia > 10 p.p. da média histórica (~27%) por 24h | Slack #ml-monitoring |
| 🔵 **Baixo** | Volume de Rejeições (422) | Pydantic rejeita > 10% das requisições em 1 hora | E-mail semanal |

---

## 3. Playbook de Incidentes

### 🔴 Incidente 1 — API Indisponível ou Modelo Não Carregado

**Cenário:** O endpoint `/api/v1/health/detailed` retorna `model: null` ou a API não responde.

| Etapa | Ação | Responsável |
| :---: | :--- | :--- |
| 1 | Verificar logs do `LoggingMiddleware` para identificar o último erro registrado (`status=5xx`). | Engenheiro de Backend |
| 2 | Verificar se o MLflow Tracking URI (`mlruns`) está acessível e se o modelo `ChurnLogisticRegression/latest` existe no Registry. | Engenheiro de MLOps |
| 3 | Se o MLflow estiver inacessível: verificar se o arquivo de fallback local (`models/logistic_model.pkl`) existe e não está corrompido (re-treinar se necessário via `make run`). | Engenheiro de MLOps |
| 4 | Reiniciar a aplicação FastAPI. Confirmar recuperação via `/api/v1/health`. | Engenheiro de Backend |
| 5 | Documentar o incidente e o tempo de indisponibilidade (MTTR). | Squad |

---

### 🟠 Incidente 2 — Latência Degradada (P95 > 500ms)

**Cenário:** A API FastAPI está respondendo acima de 500ms no P95, evidenciado pelo header `X-Process-Time`.

| Etapa | Ação | Responsável |
| :---: | :--- | :--- |
| 1 | Analisar os logs do `LoggingMiddleware` — identificar se a lentidão é generalizada ou concentrada em um path/IP específico. | Engenheiro de Backend |
| 2 | Verificar se o `model_service.py` está re-carregando o modelo do MLflow Registry a cada requisição (deveria ser carregado no `lifespan` do startup). | Engenheiro de MLOps |
| 3 | Verificar uso de CPU/Memória do servidor. Se o gargalo for volume, escalar horizontalmente (mais réplicas). | Infra/DevOps |
| 4 | Se o `RateLimitMiddleware` estiver desativado e houver abuso, ativá-lo no `main.py` (descomentar a linha no `app.add_middleware`). | Engenheiro de Backend |
| 5 | Monitorar por 15 minutos após a correção. Confirmar normalização do `X-Process-Time` nos logs. | Squad |

---

### 🟡 Incidente 3 — Data Drift Detectado

**Cenário:** O teste estatístico semanal detectou mudança significativa na distribuição da feature `MonthlyCharges` ou `Charge_Difference`.

| Etapa | Ação | Responsável |
| :---: | :--- | :--- |
| 1 | Verificar com a Engenharia de Dados se houve mudança na captura do dado (ex: valor passando em centavos ao invés de reais, ou nova regra de arredondamento). | Engenheiro de Dados |
| 2 | Validar se o `RawDataSchema` (Pandera) com `strict=True` e `coerce=True` continua passando. Se o schema estiver rejeitando dados, atualizar os `isin` e `ge/le` conforme a nova realidade. | Engenheiro de MLOps |
| 3 | **Se for erro no pipeline de dados:** solicitar correção ao time de dados. Enquanto isso, avaliar se é necessário pausar predições automáticas. | Squad |
| 4 | **Se a mudança for real** (ex: empresa alterou preços dos planos): executar `make run` para re-treinar os modelos com dados atualizados, seguido de `make eval` para validar as métricas. | Engenheiro de MLOps |
| 5 | Registrar o novo modelo no MLflow Model Registry. Confirmar que a API carrega a nova versão via `models:/ChurnLogisticRegression/latest`. | Engenheiro de MLOps |

---

### 🟡 Incidente 4 — Degradação de Performance do Modelo (Model Drift)

**Cenário:** A avaliação semanal com *ground truth* confirmado mostra que o F1-Score caiu 6% em relação ao baseline da última `evaluation_run` logada no MLflow.

| Etapa | Ação | Responsável |
| :---: | :--- | :--- |
| 1 | Consultar o MLflow UI (`make mlflow-ui`) para comparar a run degradada com a run de baseline original. Identificar quais métricas caíram mais (Recall? Precisão? AUC?). | Engenheiro de MLOps |
| 2 | Isolar a base de clientes recentes e executar EDA (Notebook ou Dashboard Streamlit, aba *Análise Exploratória*) para identificar novos perfis de clientes não mapeados no treino original. | Cientista de Dados |
| 3 | Re-treinar o pipeline completo com a base histórica atualizada: `make run` → `make eval`. Comparar as métricas do novo modelo no MLflow. | Engenheiro de MLOps |
| 4 | Validar a `estimated_savings` do novo modelo no MLflow (ou no Dashboard Streamlit, aba *Custo-Benefício*) para garantir que a economia financeira não regrediu. | Squad + Negócio |
| 5 | Promover o novo modelo para produção no MLflow Registry. Monitorar a distribuição de predições por 48h para confirmar estabilização. | Engenheiro de MLOps |

---

## 4. Ferramentas e Integrações Existentes

| Componente | Ferramenta | Papel no Monitoramento |
| :--- | :--- | :--- |
| **Logging da API** | `LoggingMiddleware` (FastAPI) | Registra método, path, status code e tempo de processamento de cada requisição. |
| **Latência** | Header `X-Process-Time` | Emitido automaticamente em cada resposta, permite agregação externa. |
| **Segurança** | `SecurityHeadersMiddleware` | Emite `X-Request-ID` para rastreabilidade de requisições individuais. |
| **Rate Limiting** | `RateLimitMiddleware` | Proteção contra abuso (100 req/min por IP). Headers `X-RateLimit-*`. |
| **Validação de Entrada** | Pydantic (`PredictInput`) | Rejeita dados fora do contrato (422) antes de chegar ao modelo. |
| **Validação de Dados Brutos** | Pandera (`RawDataSchema`) | Valida schema do CSV no início do pipeline de treino (`strict=True`). |
| **Tracking de Experimentos** | MLflow Tracking + Model Registry | Armazena parâmetros, métricas (F1, AUC, Recall, Savings) e artefatos de cada run. |
| **Health Check** | Endpoints `/health` e `/health/detailed` | Verificação de disponibilidade e status do modelo carregado. |
| **Dashboard** | Streamlit (`front/app_vis.py`) | Visualização interativa de métricas, matrizes de confusão e simulação financeira. |
| **CI/CD** | GitHub Actions (`ci_pipeline.yml`) | Executa testes automatizados a cada push/PR e salva relatório HTML de QA. |

---

**Data de Atualização:** 30 de Abril de 2026
**Responsável:** Agent \<Data\> (Auxiliando Bill)
