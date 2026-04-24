"""
Utilitários de Dados para os Módulos de Modelagem
==================================================
Este módulo fornece funções auxiliares usadas pelos scripts de treinamento
e avaliação. Ele atua como uma "ponte" conveniente entre o módulo de
preprocessing (data_prep.py) e os scripts de modelagem.

Por que este módulo existe?
----------------------------
Os scripts de treino precisam de apenas duas operações:
  1. Garantir reprodutibilidade antes de qualquer coisa (set_seed).
  2. Obter os dados prontos para o treinamento (get_data_splits).

Este módulo encapsula essas duas operações em uma interface simples,
sem que os scripts de treino precisem conhecer os detalhes internos
do pipeline de preprocessing.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from src.preprocessing.data_prep import (
    apply_feature_engineering,
    feature_selection_and_encoding,
    load_and_clean_data,
    scale_and_split,
)


def set_seed(seed: int = 42) -> None:
    """
    Configura a semente de aleatoriedade em todas as bibliotecas do projeto.

    Por que isso é importante?
    --------------------------
    Machine Learning envolve várias operações aleatórias: inicialização
    dos pesos da rede neural, embaralhamento dos dados (shuffle), divisão
    em treino/teste, etc. Sem fixar uma semente, cada execução pode produzir
    resultados ligeiramente diferentes, tornando impossível comparar experimentos
    ou reproduzir um resultado específico.

    Ao chamar set_seed(42) no início de qualquer script, garantimos que
    a sequência de operações aleatórias seja sempre a mesma.

    Bibliotecas configuradas:
    - Python (random): operações aleatórias nativas
    - NumPy: operações matriciais aleatórias
    - PyTorch (CPU e GPU): inicialização de pesos e operações da rede neural
    - PYTHONHASHSEED: hashing determinístico de objetos Python

    Args:
        seed: Valor inteiro para a semente. O valor 42 é uma convenção
              amplamente usada na comunidade de ML (sem motivo científico,
              apenas tradição). Qualquer inteiro funciona, desde que seja
              consistente entre os experimentos.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Para GPUs (não tem efeito se não houver GPU)

    # Força o PyTorch a usar algoritmos determinísticos na GPU.
    # Pode tornar o treinamento um pouco mais lento, mas garante reprodutibilidade.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Desativa otimização que introduz aleatoriedade

    # Garante que o Python não use hashing aleatório de strings/dicts,
    # o que poderia afetar a ordem de processamento de dados.
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.debug("Seed {} configurada para reprodutibilidade.", seed)


def get_data_splits(file_path: Path | str) -> tuple:
    """
    Executa o pipeline completo de preparação de dados e retorna os
    conjuntos prontos para uso nos scripts de treinamento e avaliação.

    Esta função é a interface principal entre o preprocessing e a modelagem.
    Ela orquestra internamente todas as etapas do data_prep.py:
      1. Carregamento e limpeza do CSV bruto
      2. Feature Engineering (criação de novas colunas)
      3. Seleção de features e One-Hot Encoding
      4. Divisão 70/15/15 com stratify + StandardScaler

    Importante sobre o Scaler retornado:
    ------------------------------------
    O 7º elemento do retorno é o `scaler` ajustado apenas com as estatísticas
    do conjunto de treino. O script de treinamento (train_baselines.py) é
    responsável por salvar esse scaler em disco (`models/scaler.pkl`).

    Em produção, ao receber um novo cliente para predição, o dado deve
    passar pelo MESMO scaler usado no treino — não por um novo scaler
    recalculado. Por isso, é fundamental que ele seja persistido.

    Args:
        file_path: Caminho para o CSV bruto do dataset Telco Churn.

    Returns:
        Tupla com 7 elementos na seguinte ordem:
          [0] X_train (pd.DataFrame): Features de treino — 70% dos dados.
          [1] y_train (pd.Series):    Alvo de treino (0 ou 1).
          [2] X_val   (pd.DataFrame): Features de validação — 15% dos dados.
          [3] y_val   (pd.Series):    Alvo de validação.
          [4] X_test  (pd.DataFrame): Features de teste — 15% dos dados.
          [5] y_test  (pd.Series):    Alvo de teste.
          [6] scaler  (StandardScaler): Scaler ajustado no treino. SALVE-O!

    Exemplo de uso:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = get_data_splits(path)
        # Se não precisar de todos os conjuntos, use _ para ignorar:
        X_train, y_train, _, _, _, _, scaler = get_data_splits(path)
    """
    df = load_and_clean_data(file_path)
    df = apply_feature_engineering(df)
    X, y = feature_selection_and_encoding(df)
    return scale_and_split(X, y)
