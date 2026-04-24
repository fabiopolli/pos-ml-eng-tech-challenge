"""
Treinamento dos Modelos de Baseline (Linha de Base)
====================================================
Este script treina dois modelos simples que servem como referência mínima
de performance para o projeto.

O que é um modelo de baseline?
--------------------------------
Antes de investir tempo em modelos complexos (como a Rede Neural), precisamos
estabelecer um "chão" de performance — a resposta mais simples possível para
o problema. Se o modelo avançado não superar o baseline, algo está errado.

Modelos treinados aqui:
  1. DummyClassifier: Sempre prevê a classe mais frequente (neste caso,
     "não churn"). É o modelo mais ingênuo possível. Se a acurácia for 73%,
     é porque 73% dos clientes não churnam de qualquer forma.

  2. LogisticRegression: Modelo linear clássico que aprende uma fronteira
     de decisão entre "churna" e "não churna". Muito mais inteligente que
     o Dummy, mas ainda simples e interpretável. Com class_weight='balanced',
     ele presta mais atenção nos casos de churn (classe minoritária).

Artefatos gerados em 'models/':
  - dummy_model.pkl    → Modelo Dummy serializado
  - logistic_model.pkl → Regressão Logística serializada
  - scaler.pkl         → Normalizador ajustado no treino (essencial para produção)

Execução:
    python src/models/train_baselines.py
"""

import joblib
from loguru import logger
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.models.config import PipelineConfig
from src.models.data_utils import get_data_splits, set_seed


def main(
    data_path: Path | None = None,
    models_dir: Path | None = None,
    config: PipelineConfig | None = None,
) -> None:
    """
    Treina os modelos de baseline (Dummy e Logistic Regression) e persiste
    os artefatos treinados em disco.

    Por que aceitar data_path e models_dir como parâmetros?
    -------------------------------------------------------
    Quando executado diretamente (`python train_baselines.py`), os caminhos
    padrão do projeto são usados automaticamente. Quando chamado pelos testes
    automatizados, caminhos temporários são injetados para não interferir
    nos modelos reais do projeto.

    Args:
        data_path: Caminho para o CSV bruto. Se None, usa o padrão do projeto
                   ('data/raw/Telco-Customer-Churn.csv').
        models_dir: Diretório para salvar os modelos. Se None, usa 'models/'.
        config: Configuração de hiperparâmetros. Se None, usa PipelineConfig()
                com todos os valores padrão.
    """
    # Carrega a configuração padrão se nenhuma for fornecida.
    cfg = config or PipelineConfig()

    # 1. Configuração e Reprodutibilidade
    # Deve ser a PRIMEIRA operação para garantir que tudo que vem depois
    # (splits, inicializações) seja determinístico.
    set_seed(cfg.seed)

    # Resolve os caminhos: usa os padrões do projeto se não foram informados.
    BASE_DIR = Path(__file__).resolve().parents[2]
    file_path = data_path or (BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv")
    out_dir = models_dir or (BASE_DIR / "models")
    out_dir.mkdir(exist_ok=True)  # Cria o diretório se não existir

    # 2. Carregamento de Dados
    # Apenas os dados de TREINO são usados aqui. Val e Test são ignorados
    # (representados por "_") pois os baselines não precisam de validação
    # durante o treinamento.
    logger.info("Efetuando carga e split dos dados...")
    try:
        X_train, y_train, _, _, _, _, scaler = get_data_splits(file_path)
    except FileNotFoundError:
        logger.error("Arquivo não encontrado em {}", file_path)
        return  # Encerra sem lançar exceção — o erro já foi logado

    # 3. Treinamento do Modelo Dummy
    # O DummyClassifier com strategy='most_frequent' simplesmente olha qual
    # classe aparece mais no treino e sempre prevê essa classe para qualquer
    # entrada. É completamente "burro" — não olha nenhuma feature.
    logger.info("Treinando Dummy Classifier (most_frequent)...")
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    # 4. Treinamento da Regressão Logística
    # Apesar do nome, é um classificador. Aprende uma combinação linear das
    # features que separa "churn" de "não churn". É rápido, interpretável e
    # frequentemente surpreendentemente eficaz para problemas tabulares.
    logger.info(
        "Treinando Regressão Logística (class_weight='{}', max_iter={})...",
        cfg.baseline.logistic_class_weight,
        cfg.baseline.logistic_max_iter,
    )
    lr = LogisticRegression(
        class_weight=cfg.baseline.logistic_class_weight,  # 'balanced' compensa o desbalanceamento
        max_iter=cfg.baseline.logistic_max_iter,           # Iterações máximas para convergir
        random_state=cfg.baseline.random_state,            # Para reprodutibilidade
    )
    lr.fit(X_train, y_train)

    # 5. Persistência dos Modelos e do Scaler
    # joblib.dump serializa o objeto Python para um arquivo binário (.pkl).
    # joblib.load (usado na avaliação) faz o processo inverso.
    # O scaler DEVE ser salvo aqui pois é necessário para normalizar novos
    # dados antes da predição em produção.
    joblib.dump(dummy, out_dir / "dummy_model.pkl")
    joblib.dump(lr, out_dir / "logistic_model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")

    logger.success("Artefatos salvos em '{}':", out_dir)
    logger.info("  - dummy_model.pkl")
    logger.info("  - logistic_model.pkl")
    logger.info("  - scaler.pkl")


if __name__ == "__main__":
    main()
