# Model Card — Previsão de Churn em Telecom

## 1. Visão Geral

### Modelos Documentados
- **ChurnMLP** — Rede neural PyTorch para previsão de churn.
- **ChurnLogisticRegression** — Regressão logística como baseline de produção.
- **ChurnDummyClassifier** — Baseline de referência com estratégia majoritária.

### Objetivo
Prever a probabilidade de churn de clientes de telecomunicações para suportar decisões de retenção proativa.

### Tipo de Problema
Classificação binária: `Churn = 0` (não churn) vs. `Churn = 1` (churn).

### Responsáveis
- **Fábio Polli** — Infraestrutura e baselines.
- **Willian Kopp** — Arquitetura de deep learning.
- **Romário Silva** — Definição de negócio e EDA.
- **Denis Barros Melo** — Engenharia de dados e governança.

### Data de Criação
Maio 2026

### Versão do Documento
1.0

## 2. Uso do Modelo

### Aplicações Recomendadas
- Identificação de clientes com alto risco de churn.
- Prioritização de campanhas de retenção.
- Suporte a decisões de oferta personalizada.

### Aplicações Não Recomendadas
- Uso em domínios fora de telecom (ex.: e-commerce, financeiro) sem re-treinamento.
- Aplicação em bases muito antigas ou com distribuição diferente da original.
- Decisões automáticas de alto impacto sem revisão humana.

### Consumo via API
- **Endpoint**: `POST /api/v1/predict`
- **Host padrão**: `http://localhost:8000`
- **Payload de exemplo**:
  ```json
  {
    "tenure": 12,
    "monthly_charges": 50.0,
    "total_charges": 600.0,
    "contract_type": "monthly",
    "payment_method": "credit_card",
    "has_phone_service": true,
    "has_internet_service": true,
    "has_online_security": false
  }
  ```
- **Resposta esperada**:
  ```json
  {
    "prediction": 1,
    "probability": 0.85
  }
  ```

## 3. Dados

### Fonte e Contexto
- **Dataset**: `Telco-Customer-Churn.csv`
- **Localização**: `data/raw/`
- **Origem**: Dados públicos de telecomunicações
- **Volume**: Aproximadamente 7.000 amostras

### Variáveis Utilizadas
- `tenure`
- `monthly_charges`
- `total_charges`
- `contract_type`
- `payment_method`
- `has_phone_service`
- `has_internet_service`
- `has_online_security`

### Distribuição de Classes
- `Churn = 0`: ~73%
- `Churn = 1`: ~27%
- Observação: há desbalanceamento significativo em favor da classe negativa.

### Pré-processamento
- Validação de esquema com Pandera
- Conversão de variáveis categóricas em features numéricas
- Normalização com StandardScaler
- Split em treino / validação / teste (70/15/15)
- Implementação em `src/preprocessing/data_prep.py`

## 4. Métricas de Avaliação

### Métricas Principais (última `evaluation_run` no MLflow)
- **Accuracy**: ~80%
- **F1-Score (weighted)**: ~78%
- **Recall (weighted)**: ~75%
- **AUC-ROC**: ~0.85
- **Precisão**: ~82%

### Comparativo de Modelos
| Modelo | Accuracy | F1 | Recall | AUC |
|--------|----------|-----|--------|-----|
| ChurnMLP | 81% | 79% | 76% | 0.86 |
| ChurnLogisticRegression | 79% | 77% | 74% | 0.84 |
| DummyClassifier | 73% | 60% | 50% | 0.50 |

### Curvas de Aprendizado
- **Train loss** decresce de forma consistente.
- **Validation loss** estabiliza com pouca evidência de overfitting.
- Gráficos disponíveis em MLflow UI.

### Métrica de Negócio
- **Economia estimada**: custo evitado de churn comparado ao custo de retenção.
- Esta métrica deve ser recalculada com dados de negócio reais antes de cada implantação.

## 5. Arquitetura de Modelo

### ChurnMLP
- **Framework**: PyTorch
- **Estrutura**:
  - Linear(in_dim → 32)
  - ReLU
  - Dropout(0.2)
  - Linear(32 → 16)
  - ReLU
  - Dropout(0.2)
  - Linear(16 → 1)
  - Sigmoid
- **Hiperparâmetros principais**:
  - Learning rate: 0.001
  - Batch size: 32
  - Épocas: 50
  - Otimizador: Adam
  - Loss: BCEWithLogitsLoss

### ChurnLogisticRegression
- **Framework**: scikit-learn
- **Parâmetros**: `C=1.0`, `max_iter=1000`

### ChurnDummyClassifier
- **Framework**: scikit-learn
- **Estratégia**: `most_frequent`

### Dependências principais
- PyTorch
- scikit-learn
- pandas
- numpy
- mlflow

## 6. Limitações e Riscos

### Limitações Técnicas
- Treinado especificamente para clientes de telecom; generalização para outros domínios é limitada.
- Desbalanceamento de classes pode afetar precisão para a classe minoritária.
- Ausência de variáveis comportamentais e de histórico de atendimento.
- Requer re-treinamento se a distribuição de clientes mudar significativamente.

### Vieses Identificados
- `payment_method`: clientes com `electronic_check` apresentam maior taxa de churn.
- `contract_type`: clientes em contratos mensais têm maior risco de churn.
- Risco de decisões enviesadas em campanhas de retenção quando atributos correlacionados à renda ou comportamento não forem controlados.

### Riscos de Privacidade
- Incluir `monthly_charges` e `total_charges` exige cuidado com LGPD/GDPR.
- Recomenda-se anonimização e controles de acesso aos dados de entrada e saída.

### Cenários de Falha
- Entrada fora dos limites definidos pelo schema Pydantic resulta em `HTTP 422`.
- Falha de carregamento do modelo no MLflow pode acionar fallback para o modelo local.
- Queda de performance superior a 5% em F1 deve disparar revisão e re-treinamento.

## 7. Monitoramento e Manutenção

### Monitoramento
- **MLflow Tracking** para métricas de treino, validação e teste.
- **Middleware da API** para latência e logging.
- Metas operacionais:
  - Latência P95 < 200 ms
  - Proporção de churn próxima a 27%
  - F1-Score semanal estável em relação ao baseline

### Re-treinamento
- Revisão periódica mensal ou quando houver mudanças relevantes nos dados.
- Processo sugerido:
  1. Executar `python main.py`
  2. Avaliar resultados em MLflow
  3. Promover nova versão no Model Registry

### Versionamento
- Modelos registrados e versionados no MLflow Model Registry.
- Fallback local disponível em `models/` quando o Registry não estiver acessível.

## 8. Contato

- **Documentação técnica**: `docs/guia_api_fastapi.md`, `docs/tutorial_mlflow.md`
- **Repositório**: GitHub
- **Suporte**: abrir issue no repositório para incidentes ou melhorias.
