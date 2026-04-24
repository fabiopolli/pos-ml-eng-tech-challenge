# 📊 Relatório de Avaliação: Dataset Telco Customer Churn

Este documento resume a análise técnica e de negócio do dataset `Telco-Customer-Churn.csv`. Ele serve como base para justificar as escolhas feitas no pipeline de Machine Learning.

---

## 1. Contexto do Problema
O objetivo é prever o **Churn** (cancelamento de serviço). Em termos técnicos, trata-se de um problema de **Classificação Binária**:
- **Classe 0 (Negativa):** O cliente permanece na empresa.
- **Classe 1 (Positiva):** O cliente cancelou o serviço (é quem queremos identificar).

**Dados:** 7.043 registros e 21 atributos originais.

---

## 2. Insights da Análise Exploratória (EDA)

Para que o time entenda as particularidades deste dataset, destacamos três pontos críticos:

### 2.1. O Desafio do Desbalanceamento
- **Fato:** ~73.5% dos clientes não saíram e ~26.5% saíram.
- **Impacto:** Se o modelo simplesmente "chutar" que ninguém sai, ele terá 73% de acerto, mas será inútil para o negócio. Por isso, precisamos de métricas como **Recall** e técnicas de balanceamento de peso.

### 2.2. A "Pegadinha" do TotalCharges
- **Fato:** A coluna `TotalCharges` (gastos totais) é lida inicialmente como texto.
- **Insight:** Existem valores em branco (' ') quando o `tenure` (tempo de casa) é zero. Isso ocorre com clientes que acabaram de assinar e ainda não completaram um mês. No pipeline, tratamos isso convertendo para `0`.

### 2.3. Perfil de Risco Identificado
Através da análise de correlação, identificamos que o risco de churn é maior em:
- **Contratos Mensais:** É a variável mais forte para prever a saída.
- **Pagamento via Electronic Check:** Este método tem uma taxa de churn desproporcionalmente alta.
- **Clientes Novos:** O risco de saída é altíssimo nos primeiros 12 meses.

---

## 3. Estratégia de Preparação de Dados (Data Prep)

Para que os modelos funcionem corretamente, seguimos estes pilares:

1.  **Limpeza:** Remoção do `customerID` (não ajuda a prever comportamento) e correção de tipos numéricos.
2.  **Codificação (Encoding):** Transformamos textos (Ex: "Sim", "Não", "Fibra Óptica") em números (0, 1) usando *One-Hot Encoding*. Sem isso, os modelos matemáticos não conseguem processar as informações.
3.  **Escalonamento (Scaling):** Usamos o `StandardScaler`. Como temos variáveis com escalas muito diferentes (ex: `tenure` de 0-72 vs `TotalCharges` de 0-8000), o escalonamento garante que nenhuma variável "atropele" a outra durante o aprendizado da rede neural.

---

## 4. Engenharia de Features (O "Pulo do Gato")

Criamos 5 novas variáveis para ajudar o modelo a entender o comportamento do cliente:

| Feature | O que é? | Por que criamos? |
| :--- | :--- | :--- |
| **Tenure_Bins** | Grupos por tempo de casa | Clientes com menos de 1 ano têm comportamentos muito diferentes de clientes antigos. |
| **Services_Count** | Total de serviços extras | Mede o "lock-in". Quanto mais serviços, mais difícil é para o cliente sair da operadora. |
| **Has_Family** | União de Partner/Dependents | Indica estabilidade residencial. Famílias tendem a churnar menos. |
| **Is_Electronic_Check** | Flag de risco de pagamento | Isola o comportamento de alto risco identificado na EDA para este método de pagamento. |
| **Charge_Difference** | Erro entre Total e Mensal | Ajuda a detectar se houve reajustes ou cobranças extras que geraram insatisfação. |

---

## 5. Plano de Modelagem

Dividimos nossa abordagem em duas frentes para garantir robustez:

1.  **Baselines (Simples):** Usamos Regressão Logística para ter uma métrica de comparação rápida. Se a Rede Neural não for melhor que isso, não vale a complexidade.
2.  **Deep Learning (Complexo):** Criamos uma MLP (Rede Neural) em PyTorch para capturar relações não-lineares que modelos simples podem perder.

---
**Documento atualizado em:** 23 de Abril de 2026
**Responsável:** Equipe de ML Eng (Data & Bill)
