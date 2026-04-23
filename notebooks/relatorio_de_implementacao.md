# Relatório Técnico de Implementação: Projeto Telco Churn Prediction

Este relatório detalha a arquitetura, as decisões técnicas e os conceitos de engenharia de Machine Learning (ML Eng) aplicados no desenvolvimento deste projeto. Ele foi estruturado para servir como um guia de aprendizado sobre como transicionar de uma análise exploratória para um pipeline de produção.

---

## 1. Objetivo do Projeto

*   **A) O que foi feito:** Definição do problema de negócio como uma tarefa de **Classificação Binária**.
*   **B) Porque foi feito:** O "Churn" (cancelamento) é uma das métricas mais críticas para empresas de serviço. Prever a saída de um cliente permite que a empresa tome ações proativas de retenção (como ofertas especiais), economizando o alto custo de aquisição de novos clientes.
*   **C) Como foi feito:** Utilizamos o dataset histórico da Telco para mapear o comportamento dos clientes (variáveis independentes) e rotular quem saiu ou não (variável alvo/label).

## 2. Estrutura do Projeto e Modularização (`src/`)

*   **A) O que foi feito:** O código foi extraído do Jupyter Notebook e organizado em scripts Python (.py) dentro de uma estrutura de pastas profissional: `data/`, `models/`, `src/` e `notebooks/`.
*   **B) Porque foi feito:** Notebooks são excelentes para experimentação, mas difíceis de versionar e colocar em produção. A modularização permite que o código seja reutilizável, testável e escalável. É o primeiro passo para o **MLOps**.
*   **C) Como foi feito:** 
    *   `data/`: Armazena dados brutos e processados.
    *   `models/`: Armazena os arquivos binários dos modelos treinados (.pkl, .pth).
    *   `src/`: Contém a lógica central (scripts de treino e utilitários).
    *   `app_vis.py`: Dashboard de interface com o usuário.

## 3. Utilidade e Pré-processamento (`data_utils.py`)

*   **A) O que foi feito:** Centralização de toda a lógica de limpeza de dados e divisão de datasets em um único módulo.
*   **B) Porque foi feito:** Garantir que os dados usados no treinamento, na validação e no teste passem exatamente pelas mesmas transformações. Isso evita o **Data Leakage** (vazamento de dados), onde informações do futuro "vazam" para o treino do modelo.
*   **C) Como foi feito:** 
    *   **Tratamento de Nulos:** Conversão de strings vazias em `TotalCharges` para 0.
    *   **One-Hot Encoding:** Transformação de categorias textuais (ex: "Fibra Óptica", "DSL") em colunas numéricas (0 e 1), pois modelos matemáticos não "lêem" texto.
    *   **Divisão Estratificada:** Dividimos os dados em Treino (70%), Validação (15%) e Teste (15%) mantendo a proporção de Churn em todos os pedaços.

## 4. Engenharia de Variáveis (Feature Engineering)

*   **A) O que foi feito:** Criação de colunas "artificiais" a partir das existentes para destacar padrões comportamentais.
*   **B) Porque foi feito:** O modelo nem sempre consegue captar nuances sozinho. Ao criar variáveis específicas, "ajudamos" o algoritmo a focar em padrões que o especialista humano identificou como relevantes no EDA.
*   **C) Como foi feito (Detalhamento por variável):** 

    *   **1. Tenure_Bins (Faixas de Fidelidade):** 
        *   **Como:** Usamos `pd.cut` para agrupar o tempo de contrato (`tenure`) em 5 faixas (0-12, 13-24, 25-48, 49-60, >60 meses).
        *   **Por que:** Clientes novos (0-12 meses) têm um comportamento de risco drasticamente diferente de clientes antigos. Agrupar em bins ajuda o modelo a tratar essas janelas de tempo como "estágios" de relacionamento.
    *   **2. Services_Count (Índice de Engajamento):** 
        *   **Como:** Somamos todas as vezes que o cliente possui um serviço adicional ativo (Segurança Online, Backup, etc).
        *   **Por que:** Existe o conceito de "Sticky Services" (Serviços que prendem o cliente). Quanto mais serviços um cliente assina, maior é a sua dependência da empresa e mais difícil/burocrático é o cancelamento (Lock-in).
    *   **3. Has_Family (Estabilidade Familiar):** 
        *   **Como:** Uma flag (0 ou 1) que indica se o cliente possui parceiro(a) OU dependentes.
        *   **Por que:** Clientes com família tendem a ter contratos residenciais mais estáveis e menor propensão a mudanças impulsivas de serviço em comparação a clientes solteiros.
    *   **4. Is_Electronic_Check (Fator de Risco de Pagamento):** 
        *   **Como:** Uma flag binária específica para quem usa o método "Electronic check".
        *   **Por que:** No EDA, descobrimos que este método de pagamento está correlacionado com a maior taxa de churn do dataset. Isolar essa variável dá ao modelo um sinal direto de "alto risco".
    *   **5. Charge_Difference (Anomalia de Cobrança):** 
        *   **Como:** Calculamos `TotalCharges - (MonthlyCharges * tenure)`.
        *   **Por que:** Essa conta deveria ser próxima de zero. Valores altos indicam que o cliente teve aumentos, taxas extras ou serviços adicionais no passado, o que pode gerar insatisfação financeira.

## 5. Estratégia de Modelagem (Baselines e Deep Learning)

*   **A) O que foi feito:** Treinamento comparativo de três modelos (Dummy, LogReg e MLP).
*   **B) Porque foi feito:** Precisávamos entender se a complexidade de uma rede neural (MLP) justificaria o custo computacional em relação a um modelo simples (LogReg).
*   **C) Como foi feito (O Processo de Treinamento):** 

    *   **Variáveis Eleitas (As "Features"):** Escolhemos `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, `OnlineSecurity`, `TechSupport`, `InternetService` e `PaymentMethod`, além das 5 variáveis criadas acima.
    *   **Por que estas variáveis?** Elas foram selecionadas com base no EDA original, que mostrou que o Churn é altamente influenciado por:
        1. **Tipo de Contrato:** Contratos mensais são os maiores causadores de churn.
        2. **Qualidade do Serviço:** Clientes de fibra óptica reclamam mais e saem mais.
        3. **Fidelidade:** Quanto menor o tenure, maior o risco.
    *   **Treinamento da Regressão Logística:** Configuramos o parâmetro `class_weight='balanced'`. Isso é vital porque temos poucos exemplos de Churn (26%). O modelo "pesa" mais os erros na classe minoritária para não ignorá-la.
    *   **Treinamento da Rede Neural (MLP):** 
        *   **Arquitetura:** 32 neurônios -> 16 neurônios -> 1 saída.
        *   **Processo:** Usamos o otimizador Adam e a função de perda BCE (Binary Cross Entropy). O modelo passou 50 vezes pelos dados (`epochs`), ajustando os pesos a cada erro cometido, buscando minimizar a perda na validação.

## 6. Avaliação e Métricas de Performance

*   **A) O que foi feito:** Avaliação multicriterial usando Precisão, Recall, F1-Score e análise visual via Matriz de Confusão.
*   **B) Porque foi feito (Lógica de Negócio):** No problema de Churn, os erros têm pesos diferentes. 
    *   **Erro Tipo I (Falso Positivo):** Prever que o cliente vai sair, mas ele fica. *Custo:* Dar um desconto ou brinde desnecessário.
    *   **Erro Tipo II (Falso Negativo):** Prever que o cliente fica, mas ele sai. *Custo:* Perder o valor vitalício (LTV) do cliente.
    *   **Conclusão:** É muito mais caro perder um cliente do que dar um brinde. Por isso, nossa lógica prioriza o **Recall**.
*   **C) Como foi feito (Detalhamento das Métricas):** 

    *   **1. Recall (Sensibilidade - "O Caçador de Churn"):**
        *   **Definição Profunda:** É a capacidade do modelo de encontrar **todos** os casos positivos. Matematicamente: `Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos)`.
        *   **Exemplo Prático:** Imagine que 100 clientes vão cancelar este mês. Se o modelo tem **80% de Recall**, ele "pescou" 80 desses clientes, mas deixou 20 "fugirem" sem aviso.
        *   **Nossa Lógica:** No Churn, o Recall é o rei. Queremos que esses 20 que fugiram sejam o menor número possível, mesmo que tenhamos que incomodar alguns clientes extras.
    *   **2. Precisão (Exatidão - "A Qualidade do Alarme"):**
        *   **Definição Profunda:** É a confiança que temos quando o modelo dá um alarme. Matematicamente: `Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos)`.
        *   **Exemplo Prático:** O modelo aponta 100 clientes como "vão cancelar". Se apenas 60 realmente cancelarem, sua **Precisão é de 60%**. Os outros 40 foram "falsos alarmes".
        *   **Nossa Lógica:** Se a precisão for muito baixa (ex: 5%), o time de retenção vai gastar muito tempo e dinheiro ligando para pessoas que nunca pensaram em sair. Buscamos um equilíbrio aceitável (~50-60%).
    *   **3. F1-Score (O Equilíbrio Harmônico):**
        *   **Definição Profunda:** É a média harmônica entre Precisão e Recall. Ela é usada porque "pune" valores extremos. Se um modelo tem 100% de Recall mas 0% de Precisão, o F1-Score será próximo de zero.
        *   **Por que não a média comum?** Porque se você tem 100% de Recall e 0% de Precisão, a média comum daria 50% (parecendo bom), enquanto a harmônica daria 0% (mostrando que o modelo é inútil).
        *   **Nossa Lógica:** Usamos o F1 para garantir que não estamos sacrificando toda a precisão em troca de recall, ou vice-versa.
    *   **4. Matriz de Confusão (A Tabela da Verdade):**
        *   **Análise Visual:** É uma tabela 2x2 que cruza a **Previsão** do modelo com a **Realidade**.
            1.  **Verdadeiro Negativo (TN):** O modelo disse "Fica" e o cliente ficou. (Sucesso de retenção passiva).
            2.  **Verdadeiro Positivo (TP):** O modelo disse "Sai" e o cliente saiu. (Sucesso de detecção).
            3.  **Falso Positivo (FP):** O alarme soou à toa. (Custo de operação inútil).
            4.  **Falso Negativo (FN):** O cliente saiu e o modelo não viu. (**O Quadrante mais perigoso**).
        *   **Nossa Lógica:** Olhamos para a diagonal principal (TN e TP) para ver os acertos, mas nosso foco clínico é reduzir o número de **Falsos Negativos**.

## 7. Dashboard Interativo e Monitoramento (`app_vis.py`)

*   **A) O que foi feito:** Criação de um painel visual em tempo real com suporte a **Modo Escuro (Dark Mode)**.
*   **B) Porque foi feito:** Modelos de ML não podem ser "caixas pretas". Cientistas de dados e gestores precisam ver os dados e entender o desempenho do modelo de forma intuitiva e visual.
*   **C) Como foi feito:** 
    *   Utilizamos `Streamlit` para a interface.
    *   CSS customizado para garantir uma estética premium e legibilidade.
    *   Cálculo dinâmico de métricas: ao abrir o dashboard, ele carrega os modelos salvos e re-calcula o desempenho atual.

---

**Conceitos Chave para Iniciantes:**
*   **Estratificação:** Garantir que o Churn apareça na mesma proporção em todos os sub-datasets.
*   **Normalização (Z-Score):** Ajustar os números para que variáveis grandes (como TotalCharges) não "atropelem" variáveis pequenas (como Tenure) no aprendizado.
*   **Recall:** De 100 pessoas que iam cancelar, quantas eu consegui pegar? (Vital para Churn).

---
**Data de Atualização:** 21 de Abril de 2026
**Responsável:** Agent <Data> (Auxiliando Bill)
