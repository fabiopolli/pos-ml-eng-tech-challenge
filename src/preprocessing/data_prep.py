"""
Pipeline de Preparação de Dados (Preprocessing)
================================================
Este módulo contém todas as etapas de transformação do dataset bruto de
churn até o formato final pronto para treinamento dos modelos.

O que é preprocessing (pré-processamento)?
-------------------------------------------
Modelos de Machine Learning não conseguem trabalhar diretamente com
tabelas brutas. Os dados precisam ser limpos, transformados e formatados
de uma forma específica. Este módulo faz exatamente isso, seguindo a
sequência abaixo:

  1. load_and_clean_data()        → Carrega o CSV e remove/corrige problemas
  2. apply_feature_engineering()  → Cria novas colunas com informações úteis
  3. feature_selection_and_encoding() → Seleciona as colunas relevantes e
                                        converte texto em números
  4. scale_and_split()            → Divide em treino/val/teste e normaliza

Fluxo típico de uso:
--------------------
    df = load_and_clean_data("data/raw/Telco-Customer-Churn.csv")
    df = apply_feature_engineering(df)
    X, y = feature_selection_and_encoding(df)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = scale_and_split(X, y)
"""

import pandas as pd
from loguru import logger
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocessing.schemas import RawDataSchema


def load_and_clean_data(file_path: Path | str) -> pd.DataFrame:
    """
    Carrega o dataset bruto do disco e aplica limpeza básica.

    Problemas conhecidos no dataset original que esta função corrige:
    - A coluna 'customerID' é um identificador único sem valor preditivo.
    - A coluna 'TotalCharges' vem como texto (string) e precisa ser
      convertida para número. Clientes com tenure=0 têm TotalCharges
      em branco, que são substituídos por 0.
    - A coluna 'Churn' vem como "Yes"/"No" e é mapeada para 1/0
      (o formato que os algoritmos esperam).

    Args:
        file_path: Caminho completo para o arquivo CSV do dataset.

    Returns:
        DataFrame limpo, sem a coluna customerID, com TotalCharges
        numérico e Churn como 0 ou 1.

    Raises:
        FileNotFoundError: Se o arquivo não existir no caminho informado.
    """
    logger.info("Carregando dados de: {}", file_path)
    df = pd.read_csv(file_path)

    # Validação rigorosa com Pandera (Fail Fast)
    logger.info("Validando esquema dos dados brutos com Pandera...")
    df = RawDataSchema.validate(df)
    logger.success("Dados brutos validados com sucesso!")

    # Remove o ID do cliente — é um identificador único que não tem valor
    # preditivo (o modelo não deve aprender com o número do cliente).
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # TotalCharges vem como texto no CSV original. Clientes novos (tenure=0)
    # têm essa coluna em branco, o que seria lido como NaN. Substituímos
    # por 0, que é o valor correto (eles ainda não pagaram nada).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # O modelo precisa de números, não de texto. "Yes" → 1, "No" → 0.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    logger.debug("Dataset carregado: {} registros, {} colunas", *df.shape)
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas colunas (features) a partir das existentes para enriquecer
    o conjunto de dados com informações que podem melhorar as predições.

    O que é Feature Engineering?
    -----------------------------
    Em vez de usar apenas as colunas originais do dataset, podemos criar
    novas combinações que capturam padrões relevantes. Por exemplo:
    em vez de só saber o 'tenure' (tempo de cliente), agrupamos em faixas
    (0-12 meses, 1-2 anos, etc.) para capturar comportamentos por ciclo de vida.

    Novas colunas criadas:
        Tenure_Bins: Faixa de tempo como cliente (ex: "0-12", "13-24", ">60").
        Services_Count: Total de serviços opcionais contratados.
        Has_Family: 1 se tem cônjuge ou dependentes, 0 caso contrário.
        Is_Electronic_Check: 1 se paga por débito eletrônico (meio de pagamento
            associado a clientes com maior taxa de churn no EDA).
        Charge_Difference: Diferença entre o total cobrado e o esperado
            (MonthlyCharges × tenure). Diferenças grandes podem indicar
            mudanças de plano ou cobranças extras.

    Args:
        df: DataFrame já limpo por load_and_clean_data().

    Returns:
        O mesmo DataFrame com as 5 novas colunas adicionadas.
    """
    logger.info("Aplicando Feature Engineering...")

    # Agrupa o tempo de cliente em faixas. Clientes com <12 meses são muito
    # diferentes de clientes com >5 anos — esse agrupamento captura isso.
    df["Tenure_Bins"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 100],
        labels=["0-12", "13-24", "25-48", "49-60", ">60"],
    )

    # Conta quantos serviços adicionais o cliente assinou. Clientes com
    # mais serviços tendem a ter mais "lock-in" e menor taxa de churn.
    services_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["Services_Count"] = (df[services_cols] == "Yes").sum(axis=1)

    # Proxy de estabilidade familiar. Clientes com família tendem a
    # priorizar a estabilidade do serviço e churnar menos.
    df["Has_Family"] = (
        (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
    ).astype(int)

    # Flag para o método de pagamento "débito eletrônico". O EDA identificou
    # esse grupo como de alto risco de churn — vale criar uma feature dedicada.
    df["Is_Electronic_Check"] = (
        df["PaymentMethod"] == "Electronic check"
    ).astype(int)

    # Mede se o total cobrado está alinhado com o esperado (custo mensal × meses).
    # Valores negativos indicam que o cliente pagou MENOS que o esperado
    # (possíveis descontos ou períodos gratuitos).
    df["Charge_Difference"] = df["TotalCharges"] - (
        df["MonthlyCharges"] * df["tenure"]
    )

    return df


def feature_selection_and_encoding(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Seleciona as colunas mais relevantes e converte variáveis categóricas
    em formato numérico para que os modelos possam processá-las.

    Por que selecionar features?
    ----------------------------
    Nem todas as colunas do dataset são úteis para prever churn. Incluir
    colunas irrelevantes pode confundir o modelo (ruído) e aumentar o custo
    computacional desnecessariamente. As features selecionadas foram definidas
    com base na análise exploratória (EDA) documentada em /docs.

    O que é One-Hot Encoding?
    --------------------------
    Colunas categóricas como 'Contract' têm valores como "Month-to-month",
    "One year", "Two year". O modelo não entende texto, por isso criamos
    uma coluna binária (0 ou 1) para cada categoria. O parâmetro drop_first=True
    remove uma das categorias para evitar multicolinearidade.

    Exemplo de encoding da coluna 'Contract':
      Antes: "Month-to-month"
      Depois: Contract_One year=0, Contract_Two year=0
              (a ausência de ambos implica Month-to-month)

    Args:
        df: DataFrame com as features originais + as novas de engenharia.

    Returns:
        Tupla (X, y) onde:
        - X (pd.DataFrame): Matriz de features prontas para o modelo.
        - y (pd.Series): Coluna alvo com 1 (churn) ou 0 (não churn).
    """
    logger.info("Realizando Seleção e Codificação de Features...")

    # Separamos o alvo (y) do restante das features (X) antes de qualquer
    # transformação para não vazar informação do target no pré-processamento.
    target = df["Churn"]

    # Lista das colunas selecionadas para treinamento. Inclui features
    # originais de alto poder preditivo + as 5 criadas na etapa anterior.
    critical_features = [
        "tenure", "Contract", "MonthlyCharges", "TotalCharges",
        "OnlineSecurity", "TechSupport", "InternetService", "PaymentMethod",
        "Tenure_Bins", "Services_Count", "Has_Family", "Is_Electronic_Check",
        "Charge_Difference",
    ]

    features = df[critical_features]

    # Identifica automaticamente colunas de texto/categoria para aplicar OHE.
    # As colunas numéricas (tenure, MonthlyCharges, etc.) são mantidas como estão.
    cat_cols = features.select_dtypes(include=["object", "category"]).columns

    # pd.get_dummies: aplica o One-Hot Encoding.
    # drop_first=True: remove a primeira categoria de cada variável para
    # evitar multicolinearidade (o chamado "dummy variable trap").
    # dtype=int: garante que as colunas binárias sejam inteiros (0/1),
    # não booleanos, o que é mais compatível com sklearn e PyTorch.
    features = pd.get_dummies(features, columns=cat_cols, drop_first=True, dtype=int)

    logger.debug("Features após encoding: {} colunas", features.shape[1])
    return features, target


def scale_and_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Divide o dataset em Treino, Validação e Teste, e normaliza as features
    numéricas usando o StandardScaler (Z-Score).

    Por que dividir em 3 conjuntos?
    --------------------------------
    - Treino (70%): O modelo aprende com esses dados.
    - Validação (15%): Usado durante o treinamento para monitorar se o modelo
      está generalizando ou apenas memorizando (overfitting). A MLP usa isso
      a cada época para exibir a loss de validação.
    - Teste (15%): Dados "nunca vistos" usados APENAS na avaliação final para
      medir o desempenho real do modelo em produção.

    Por que usar stratify?
    ----------------------
    O dataset é desbalanceado: ~73% não churn e ~27% churn. Sem stratify,
    por azar, um split poderia concentrar quase todos os casos de churn em
    um único conjunto. stratify=y garante que a proporção 73%/27% seja
    mantida em treino, validação e teste.

    Por que normalizar apenas com as estatísticas do treino?
    --------------------------------------------------------
    O StandardScaler calcula a média e o desvio padrão para "centralizar"
    os dados (média≈0, std≈1). Se calcularmos essas estatísticas com base
    em todos os dados (incluindo val e teste), estamos "vazando" informação
    futura para o treinamento — o chamado Data Leakage. A regra é:
      - fit_transform() só no treino (calcula e aplica).
      - transform() no val e teste (aplica com as estatísticas do treino).

    Args:
        X: Matriz de features codificadas (saída de feature_selection_and_encoding).
        y: Coluna alvo binária (0/1).
        val_size: Proporção do total para validação. Padrão: 15%.
        test_size: Proporção do total para teste. Padrão: 15%.
        seed: Semente de aleatoriedade para reprodutibilidade dos splits.

    Returns:
        Tupla com 7 elementos:
        (X_train, y_train, X_val, y_val, X_test, y_test, scaler)

        O `scaler` DEVE ser salvo em disco pelo chamador para ser reutilizado
        na inferência (predição de novos dados em produção).
    """
    # Calculamos a proporção do "temporário" (val + test) em relação ao total.
    # Ex: val_size=0.15 + test_size=0.15 = 0.30 → 30% ficam fora do treino.
    temp_size = val_size + test_size  # 0.30

    # Primeiro split: 70% treino, 30% temporário.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=seed, stratify=y
    )

    # Segundo split: divide o temporário ao meio → 15% val, 15% teste.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )

    logger.info(
        "Split: treino={}, val={}, teste={}",
        len(X_train), len(X_val), len(X_test),
    )

    # Inicializa o scaler. Ele aprenderá a média e o desvio padrão APENAS
    # do conjunto de treino para evitar data leakage.
    scaler = StandardScaler()

    # Colunas numéricas contínuas que precisam de normalização.
    # Colunas binárias (0/1) criadas pelo OHE NÃO precisam ser escaladas.
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Services_Count", "Charge_Difference"]

    # fit_transform: calcula média/std do TREINO e já aplica a transformação.
    X_train = X_train.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

    # transform (sem fit): aplica a transformação usando as estatísticas
    # do TREINO. Isso simula o comportamento em produção, onde não temos
    # acesso aos dados futuros ao normalizar.
    X_val = X_val.copy()
    X_val[num_cols] = scaler.transform(X_val[num_cols])

    X_test = X_test.copy()
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def main() -> None:
    """
    Executa o pipeline de preprocessing de forma autônoma e salva os
    conjuntos de dados processados em 'data/processed/'.

    Use este script diretamente quando quiser inspecionar os dados
    transformados antes de treinar os modelos, ou para exportar os
    CSVs para uso em outras ferramentas (ex: notebooks, Excel).

    Execução:
        python src/preprocessing/data_prep.py
    """
    BASE_DIR = Path(__file__).resolve().parents[2]
    file_path = BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv"

    df = load_and_clean_data(file_path)
    df = apply_feature_engineering(df)
    X, y = feature_selection_and_encoding(df)

    logger.info("Dividindo o dataset (70% Treino / 15% Validação / 15% Teste)...")
    X_train, y_train, X_val, y_val, X_test, y_test, _ = scale_and_split(X, y)

    logger.info("X_train: {} | X_val: {} | X_test: {}", X_train.shape, X_val.shape, X_test.shape)
    logger.info(
        "Distribuição Churn no Treino — Não: {:.1f}% | Sim: {:.1f}%",
        y_train.value_counts(normalize=True)[0] * 100,
        y_train.value_counts(normalize=True)[1] * 100,
    )

    # Salva os conjuntos separados para facilitar análises externas.
    # Nota: o scaler não é salvo aqui pois este script é apenas exploratório.
    # O scaler para produção é salvo pelo train_baselines.py.
    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    logger.success("Conjuntos salvos em: {}", output_dir)


if __name__ == "__main__":
    main()
