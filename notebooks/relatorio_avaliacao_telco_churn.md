# Relatório de Avaliação do Dataset Telco Customer Churn

## 1. Visão Geral dos Dados
Este relatório apresenta uma análise do dataset `Telco-Customer-Churn.csv` baseada na Análise Exploratória de Dados (EDA) prévia no notebook `EDA.ipynb`. O objetivo é avaliar a qualidade dos dados contidos e prepará-los para o treinamento e inferência de modelos de Machine Learning (ML).

A base contém **7043 registros (clientes)** e **21 atributos de dados**. Pelo aspecto comportamental, o objetivo preditivo central (variável alvo) a ser modelado será a coluna **`Churn`**, indicando se o cliente cancelou (Yes) ou reteve (No) o seu serviço no último mês. Isso define o escopo empírico de um clássico problema de **Classificação Binária**.

## 2. Análise Exploratória e Principais Insights
- **Valores Ausentes (Missing Values):** Durante a avaliação superficial das matrizes foi denotada uma ausência de campos nulos. No entanto, é empiricamente notório na variável `TotalCharges` (Gastos Totais) neste dataset a presença pontual de valores em branco (' ') correspondentes aos clientes recentes, cujo tempo de contrato (`tenure`) ainda equivale a zero.
- **Tipos de Variáveis Preditoras:**
  * Há uma parcela essencialmente numérica (`tenure` medido em meses, e `MonthlyCharges`).
  * A variável `TotalCharges` usualmente costuma ser importada de arquivos .csv no formato string (`object`) pela natureza de campos irregulares descrita antes. Ela requisitará uma adaptação forçada (casting numérico).
  * A premissa dominante neste dataset é originária por classes e categorias. Atributos logísticos demográficos (`gender`, `Partner`, `Dependents`), e serviços habilitados (`OnlineSecurity`, `Contract`, `PaymentMethod`, etc) representam um corpo estribado sobre lógica unária categórica.
  * O registro `customerID` contém o identificador unitário de cada indivíduo e não sustenta peso em detecção algorítmica de comportamentos correlacionais.
- **Desbalanceamento da Distribuição Alvo:** O problema principal com o target exibe assimetria substancial para os cenários. Detectou-se que cerca de **73.5%** dos clientes permanecem retidos à empresa (classe Negativa), defrontando limitados **26.5%** que de fato compõem clientes perdidos à concorrência (Classe Positiva propensa).

## 3. Preparação de Dados Recomendada para Modelagem (Data Prep)
Para transladar este dataset rumo ao ciclo de ingestão de um pipeline produtivo de ML, os seguintes passos restam fortemente imperativos:

### 3.1. Tratamento e Limpeza (Data Cleaning)
1. **Eliminar Variáveis de Cardinalidade Inútil:** A coluna tabular `customerID` deve obrigatoriamente sofrer *drop* da matriz.
2. **Conversão Limpa (Casting):** Consertar a tipagem da coluna `TotalCharges` forçando para métricas decibais, via parsing (ex: funções como `pd.to_numeric` no Pandas). Os resquícios nulos detectados ali podem ou ser preenchidos sumariamente por `0`, refletindo ausência da cobrança decorrente do tempo de uso nulo provado por `tenure` naqueles respectivos registros, ou pode-se eliminar tais linhas.

### 3.2. Codificando Propriedades (Encoding Variables)
É imperioso estruturar o comportamento textual das variáveis para as inferências numéricas nos otimizadores e grafos do modelo preditor:
1. **Target ou Dummies Binários Simples:** Conversões puristas mapeando as features como unárias e a classe final alvo de Churn para subordinações de peso lógico `0` (No/Ausente) e `1` (Yes/Correto).
2. **Variáveis Categóricas com Multiclasse Não Ordinal (One-Hot Encoding):** Aplicável nas frentes como o formato de contrato do cliente (`Contract` com opções anuais, mensais ou binauis) ou a rotina de transações (`PaymentMethod`). Aqui usam-se técnicas de *Enconding de Variáveis Categóricas Expandidas* (get_dummies) a fim de criar as features falsas. 

### 3.3. Engenharia de Escalas de Dispersão (Feature Scaling)
A divergência escalar imposta sobre variações longínquas, contrapondo parcelas modestas nas faturas unidas (`MonthlyCharges`) contra totais anuais grandiosos (`TotalCharges`), necessitará ser controlada.
- Um regresso e redimensionamento central na forma de normalizantes por método Z-Score (usando o `StandardScaler` do módulo sklearn). **Atenção:** Este passo é crítico para o treinamento da Regressão Logística e da rede MLP (PyTorch).

### 3.4. Manejo com o Desbalanceamento Empírico
É recomendável aderir soluções profiláticas como pesos por penalidade (`class_weight='balanced'`) no Scikit-Learn ou técnicas de Oversampling (SMOTE).

### 3.5. Sugestões Avançadas de Transformação (Feature Engineering)
A criação de novas variáveis a partir das existentes ajuda modelos de Regressão Logística e MLPs a capturarem relacionamentos não-lineares e lógicas de interação de forma muito mais coesa. As sugestões analisadas e recomendadas para este dataset:
1. **Agrupamento de `tenure` (Tenure Bins):** Ao criar uma variável categórica ou ordinal que agrupa o tempo de permanência em anos ou recortes (ex: `0-12 meses`, `13-24`, `25-48`, etc.), modelamos o fato de que a curva de cancelamento decai não-linearmente (clientes super-novos têm churn altíssimo e clientes antigos se estabilizam e retém).
2. **Contagem de Serviços Opcionais (Services Count):** Compilar de maneira numérica incremental a quantidade de serviços extras ativos sob posse do cliente (ex: somar `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`). Uma feature que flutue logicamente de 0 a 6 serviços entrega muito rapidamente à MLP a métrica do nível de dependência ou "lock-in" do cliente no ecossistema da tele.
3. **Agrupamento Demográfico Familiar:** Fundir os atributos atômicos de `Partner` e `Dependents` em uma só variável de síntese `Has_Family` (booleana contendo verdadeiro caso hajam laços, falso para sem laços) consolida a indicação empírica de estabilidade daquele CPF sobre as contas do provedor num único nodo.
4. **Flag Unária de Tipo Pagamento:** Elevar uma variável de flag baseada na conversão: `Is_Electronic_Check` (como Verdedeiro ou Falso). O método demográfico "Electronic Check" costuma deter isoladamente uma assimetria desproporcional pesando nos registros que deram Churn positivo em relação aos que assinaram fatura automática via cartão de crédito.
5. **Cálculo de Discrepância Monetária:** Gerar uma feature algorítmica como `Charge_Difference = TotalCharges - (MonthlyCharges * tenure)`. Variações agudas nesse montante costumam apontar facilmente para um cliente que deixou um cenário bonificado/trial recentemente sofrendo um reajuste agudo de tarifário. Sinais deste tipo agem como triggers determinísticos para impulsionar modelos neurais (MLPs) nas ativações da camada final de output probabilístico.

## 4. Seleção de Features Críticas
Para otimizar o treinamento, devemos focar nas seguintes features, que historicamente apresentam maior correlação com o Churn neste dataset:
- **`tenure`**: Indicador direto de fidelidade; quanto menor o tempo, maior a probabilidade de churn.
- **`Contract`**: Talvez a feature mais forte; clientes com contratos "Month-to-month" têm churn drasticamente superior.
- **`MonthlyCharges`**: Indica o peso financeiro atual sobre o cliente.
- **`TotalCharges`**: Ajuda a identificar o valor de vida do cliente (LTV).
- **`OnlineSecurity`** e **`TechSupport`**: Serviços que, quando ausentes, facilitam a saída do cliente.
- **`InternetService`**: Clientes com "Fiber optic" tendem a apresentar maiores taxas de churn neste dataset (possível indicador de insatisfação com performance ou custo).
- **`PaymentMethod`**: Especialmente o método "Electronic check", frequentemente associado a taxas de churn elevadas.

## 5. Estratégia de Modelagem e Experimentação

### 5.1. Baselines (Scikit-Learn)
Serão treinados dois modelos iniciais para estabelecer referências de performance:
1.  **`DummyClassifier`**: Utilizado para definir uma linha de base "ingênua" (ex: sempre prevendo a classe majoritária), permitindo avaliar se os modelos reais estão de fato aprendendo padrões úteis.
2.  **`LogisticRegression`**: Modelo linear fundamental para classificação binária. Servirá como primeiro comparativo real de performance após o pré-processamento (Scaling e Encoding).

### 5.2. Redes Neurais (PyTorch)
Será construída uma arquitetura **MLP (Multi-Layer Perceptron)** utilizando PyTorch:
- **Camadas**: Pelo menos duas camadas ocultas com funções de ativação ReLU ou LeakyReLU.
- **Camada de Saída**: Camada linear com 1 neurônio e ativação Sigmoid para probabilidade de classe.
- **Otimização**: Uso de *Binary Cross-Entropy Loss* (BCELoss) e otimizador Adam ou SGD.
- **Regularização**: Implementação de *Dropout* ou *Weight Decay* (L2) para evitar overfitting, dado que o dataset é de tamanho moderado.

## 6. Sugestões Adicionais (Ensembles)
Consoante à literatura, após os baselines e MLP, recomenda-se explorar modelos de *Boosting* como **XGBoost** ou **CatBoost** para tentativa de superar os resultados da MLP em dados tabulares.
