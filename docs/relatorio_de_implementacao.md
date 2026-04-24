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
2.  **F1-Score:** Usado para garantir que não estamos sendo "extremos demais". Ele equilibra a precisão com o recall.
3.  **Matriz de Confusão:** Ferramenta visual para o time de negócio entender onde o modelo está errando (ex: "O modelo está sendo conservador demais ou alarmista demais?").

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

## 8. Ferramentas Utilizadas

- **UV:** Gerenciador de pacotes ultra-rápido para garantir que todos no time usem as mesmas versões de biblioteca.
- **MLflow:** Gestão de experimentos, métricas e registro de modelos.
- **Loguru:** Substituímos o `print` por logs estruturados, facilitando o debug em ambientes de produção.
- **Pathlib:** Garante que o código funcione tanto em Windows quanto Linux/Mac, tratando caminhos de arquivos de forma inteligente.

---
**Data de Atualização:** 24 de Abril de 2026
**Responsável:** Agent <Data> (Auxiliando Bill)
