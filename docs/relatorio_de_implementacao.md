# 🛠️ Relatório de Implementação Técnica: Pipeline de Churn

Este documento detalha as decisões de engenharia e a arquitetura do código. Ele serve como guia para desenvolvedores que precisam realizar manutenção ou evoluir o sistema.

---

## 1. Arquitetura do Projeto (Modularização)

**O que foi feito:** Migramos o código experimental (Notebooks) para um pacote Python estruturado (`src/`).

**Por que foi feito:**
- **Versionamento:** Scripts `.py` são mais fáceis de comparar no Git.
- **Produção:** Sistemas de nuvem e APIs executam arquivos Python, não notebooks.
- **Organização:** Separar "Preparo de Dados" de "Treinamento" permite que múltiplos desenvolvedores trabalhem em partes diferentes sem conflitos.

**Como foi feito:**
- `src/preprocessing/`: Lógica de ETL e limpeza.
- `src/models/`: Definição de arquiteturas neurais, utilitários de treino e configurações.
- `makefile`: Automação de tarefas (instalação, treino, testes) usando `uv`.

---

## 2. Fluxo de Dados e Prevenção de Data Leakage

**O que foi feito:** Centralizamos o pré-processamento no módulo `data_prep.py` e garantimos uma divisão rigorosa de datasets.

**Por que foi feito:**
- Evitar o **Data Leakage** (Vazamento de Dados): O modelo nunca deve "ver" estatísticas do conjunto de teste durante o treino. Por isso, o `StandardScaler` é ajustado (*fit*) apenas no treino e aplicado (*transform*) nos demais conjuntos.

**Como foi feito:**
- Criamos a função `get_data_splits()` que retorna 7 elementos: os 3 pares de dados (Treino, Val, Teste) e o objeto `scaler` treinado.
- **Persistência:** O `scaler.pkl` é salvo junto com os modelos para garantir que a aplicação web (Streamlit) use exatamente a mesma normalização que foi usada no treino.

---

## 3. Gestão de Configuração e Hiperparâmetros

**O que foi feito:** Criamos um arquivo `src/models/config.py` usando `dataclasses`.

**Por que foi feito:**
- Evitar o uso de "valores mágicos" (números soltos no meio do código).
- Facilitar experimentos: se quisermos testar 100 épocas em vez de 50, alteramos em um único lugar.

---

## 4. Estratégia de Testes Automatizados

**O que foi feito:** Implementamos uma suíte de testes com `pytest` cobrindo desde unidades de código até o pipeline completo (*Smoke Tests*).

**Por que foi feito:**
- Garantir que uma alteração no pré-processamento não quebre o treinamento da rede neural.
- Validar se as transformações matemáticas estão mantendo a integridade dos dados.

**Como executar:**
- `make test`: Roda os testes de lógica rápidos.
- `make test-full`: Roda os testes de integração (treina os modelos com dados sintéticos).

---

## 5. Entendendo as Métricas de Sucesso

Para este projeto, a escolha das métricas é guiada pelo **impacto financeiro**:

1.  **Recall (Métrica Alvo):** Nossa prioridade. Queremos identificar o maior número possível de clientes que vão sair. É melhor dar um desconto para quem não ia sair (Falso Positivo) do que perder um cliente por não tê-lo identificado (Falso Negativo).
2.  **AUC-ROC (Métrica Técnica):** Adicionada para medir a capacidade do modelo de distinguir entre as classes (Churn vs. Não Churn) em diferentes limiares. É uma métrica de robustez que complementa o F1-Score.
3.  **F1-Score:** Usado para garantir que não estamos sendo "extremos demais". Ele equilibra a precisão com o recall.
4.  **Matriz de Confusão:** Ferramenta visual para o time de negócio entender onde o modelo está errando (ex: "O modelo está sendo conservador demais ou alarmista demais?").

---

## 6. Treinamento Robusto da Rede Neural (PyTorch)

**O que foi feito:** Implementamos Early Stopping, Batching e Validação Cruzada Estratificada no `train_mlp.py`.

**Por que foi feito:**
- **Generalização:** O **Early Stopping** evita o overfitting, interrompendo o treino quando a perda de validação para de cair, restaurando automaticamente o melhor estado dos pesos.
- **Estabilidade:** A **Validação Cruzada Estratificada (K-Fold)** garante que a performance do modelo seja consistente em diferentes fatias dos dados, fornecendo uma estimativa de erro mais confiável (`cv_mean_val_loss`).
- **Eficiência:** O uso de **DataLoaders** com batching otimiza o uso de memória e a convergência do gradiente.

---

---

## 7. Gestão do Ciclo de Vida com MLflow

**O que foi feito:** Integramos o MLflow Tracking e Model Registry ao pipeline.

**Por que foi feito:**
- **Rastreabilidade:** Cada treinamento gera um registro único de parâmetros e métricas.
- **Comparabilidade:** Permite comparar visualmente a evolução da perda (loss) da Rede Neural entre diferentes experimentos.
- **Versionamento de Modelos:** Centralizamos os modelos no Registry, permitindo transições de estado (Staging/Production).

**Como foi feito:**
- **Tracking:** Parâmetros e métricas são logados automaticamente em `mlruns/`.
- **Model Registry:** Modelos são registrados com nomes como `ChurnMLP` e `ChurnLogisticRegression`.
- **Isolamento em Testes:** Criamos fixtures para garantir que os testes automatizados não poluam o histórico de experimentos reais.

---

## 8. Análise de Custo-Benefício e Impacto de Negócio

**O que foi feito:** Criamos uma integração entre as métricas de erro do modelo e o impacto financeiro real.

**Por que foi feito:**
- **Trade-off de Erros:** Um Falso Negativo (perder um cliente) costuma ser muito mais caro que um Falso Positivo (custo de uma ação de retenção).
- **Dashboard Interativo:** Permitimos que o time de negócio simule diferentes cenários de Ticket Médio e Custo de Campanha no Streamlit para ver a economia real gerada por cada modelo.

**Como foi feito:**
- Implementamos o cálculo de **Economia Estimada (Savings)** baseado na Matriz de Confusão.
- Logamos a métrica `estimated_savings` no MLflow para cada run de avaliação.
- Adicionamos uma aba de simulação financeira no Dashboard Streamlit.

---

## 9. Validação de Dados com Pandera

**O que foi feito:** Implementamos uma camada de validação de esquema (Data Schema) usando Pandera no início do pipeline de pré-processamento.

**Por que foi feito:**
- **Fail Fast:** Detectar inconsistências nos dados brutos (tipos errados, colunas ausentes, valores fora do esperado) antes de iniciar o treinamento ou a engenharia de features.
- **Integridade:** Garantir que o contrato entre a fonte de dados e o modelo seja respeitado.

**Como foi feito:**
- Criamos o arquivo `src/preprocessing/schemas.py` com a definição do `RawDataSchema`.
- Integramos a validação na função `load_and_clean_data` do `data_prep.py`.

---

## 10. Disponibilização via API (FastAPI)

**O que foi feito:** Criamos uma API REST robusta para servir as predições de churn em tempo real.

**Por que foi feito:**
- **Consumo:** Permite que outros sistemas (mobile, CRM, web) consultem o modelo via HTTP.
- **Padronização:** Uso de Pydantic para garantir que as entradas e saídas sigam um contrato rigoroso.

**Como foi feito:**
- **Arquitetura:** Organizada em rotas (`/predict`), serviços e esquemas de dados.
- **Prevenção de Skew:** Implementamos o `inference_preprocessor.py` que reutiliza a lógica oficial do `data_prep.py`. Isso garante que as transformações de engenharia de features e o One-Hot Encoding sejam idênticos ao que foi visto no treinamento, evitando erros de predição.
- **Registry Integration:** A API tenta carregar o modelo mais recente (`latest`) diretamente do MLflow Model Registry, com fallback automático para o arquivo local se necessário.

---

## 11. Ferramentas Utilizadas

- **UV:** Gerenciador de pacotes ultra-rápido para garantir que todos no time usem as mesmas versões de biblioteca.
- **FastAPI:** Framework moderno e de alta performance para construção de APIs.
- **MLflow:** Gestão de experimentos, métricas e registro de modelos.
- **Pandera:** Validação estatística e de esquema para DataFrames.
- **Loguru:** Substituímos o `print` por logs estruturados, facilitando o debug em ambientes de produção.
- **Pathlib:** Garante que o código funcione tanto em Windows quanto Linux/Mac, tratando caminhos de arquivos de forma inteligente.

---
**Data de Atualização:** 25 de Abril de 2026
**Responsável:** Agent <Data> (Auxiliando Bill)
