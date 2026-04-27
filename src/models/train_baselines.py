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

Integração com MLflow:
-----------------------
Cada execução deste script cria uma "run" no experimento "churn-prediction"
do MLflow, registrando automaticamente:
  - Parâmetros: hiperparâmetros do BaselineConfig (max_iter, class_weight...)
  - Métricas: acurácia dos dois modelos no conjunto de treino
  - Modelos: os dois modelos sklearn logados com assinatura de I/O e
             registrados no MLflow Model Registry para versionamento

Artefatos gerados em 'models/' (compatibilidade legada):
  - dummy_model.pkl    → Modelo Dummy serializado
  - logistic_model.pkl → Regressão Logística serializada
  - scaler.pkl         → Normalizador ajustado no treino (essencial para produção)

Execução:
    python src/models/train_baselines.py
"""

import os
import joblib
import mlflow
import mlflow.sklearn
from loguru import logger
from mlflow.models.signature import infer_signature
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.models.config import PipelineConfig
from src.models.data_utils import get_data_splits, set_seed


def main(
    data_path: Path | None = None,
    models_dir: Path | None = None,
    config: PipelineConfig | None = None,
) -> None:
    """
    Treina os modelos de baseline (Dummy e Logistic Regression) e persiste
    os artefatos treinados em disco e no MLflow Model Registry.

    Por que aceitar data_path e models_dir como parâmetros?
    -------------------------------------------------------
    Quando executado diretamente (`python train_baselines.py`), os caminhos
    padrão do projeto são usados automaticamente. Quando chamado pelos testes
    automatizados, caminhos temporários são injetados para não interferir
    nos modelos reais do projeto.

    O MLflow nos testes:
    --------------------
    Nos testes automatizados, o tracking URI é redirecionado para um diretório
    temporário (via `mlflow.set_tracking_uri(tmp_path)` no conftest.py),
    garantindo que os runs de teste não poluam o experimento de desenvolvimento.

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

    # =========================================================================
    # 3. Configuração do MLflow
    # =========================================================================
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(cfg.mlflow.experiment_name)
 
    # mlflow.start_run() abre um contexto de rastreamento. Tudo que for logado
    # (params, metrics, modelos) dentro deste bloco pertence a esta "run".
    # run_name é um rótulo amigável visível na UI do MLflow.
    with mlflow.start_run(run_name="baseline_run", nested=True):

        # --- Log de Parâmetros ---
        # mlflow.log_params registra os hiperparâmetros como pares chave-valor.
        # Estes são os "botões" que configuramos ANTES do treinamento.
        # Na UI do MLflow, você pode comparar runs com diferentes parâmetros
        # para entender o impacto de cada configuração.
        mlflow.log_params({
            "seed": cfg.seed,
            "logistic_max_iter": cfg.baseline.logistic_max_iter,
            "logistic_class_weight": cfg.baseline.logistic_class_weight,
            "random_state": cfg.baseline.random_state,
        })

        # 4. Treinamento do Modelo Dummy
        # O DummyClassifier com strategy='most_frequent' simplesmente olha qual
        # classe aparece mais no treino e sempre prevê essa classe para qualquer
        # entrada. É completamente "burro" — não olha nenhuma feature.
        logger.info("Treinando Dummy Classifier (most_frequent)...")
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)

        # 5. Treinamento da Regressão Logística
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

        # --- Log de Métricas de Treino ---
        # Registramos a acurácia no conjunto de treino como uma métrica de
        # sanidade — não é a métrica final (essa fica no evaluate_models.py),
        # mas nos ajuda a confirmar que os modelos foram treinados corretamente.
        dummy_train_acc = accuracy_score(y_train, dummy.predict(X_train))
        lr_train_acc = accuracy_score(y_train, lr.predict(X_train))

        mlflow.log_metrics({
            "dummy_train_accuracy": dummy_train_acc,
            "lr_train_accuracy": lr_train_acc,
        })
        logger.info(
            "Métricas de treino | Dummy Acc: {:.4f} | LR Acc: {:.4f}",
            dummy_train_acc, lr_train_acc,
        )

        # --- Log dos Modelos no MLflow ---
        # infer_signature analisa X_train e as predições para gerar automaticamente
        # o "contrato" de I/O do modelo: quais features ele espera e o que retorna.
        # Isso é essencial para validação em produção e para o Serving do MLflow.
        dummy_signature = infer_signature(X_train, dummy.predict(X_train))
        lr_signature = infer_signature(X_train, lr.predict(X_train))

        # mlflow.sklearn.log_model salva o modelo serializado como um artefato
        # da run, junto com dependências e metadados. O artifact_path é o nome
        # da subpasta dentro dos artefatos da run na UI do MLflow.
        dummy_model_info = mlflow.sklearn.log_model(
            sk_model=dummy,
            artifact_path="dummy_model",
            signature=dummy_signature,
            registered_model_name="ChurnDummyClassifier" if cfg.mlflow.register_model else None,
        )
        lr_model_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="logistic_model",
            signature=lr_signature,
            registered_model_name="ChurnLogisticRegression" if cfg.mlflow.register_model else None,
        )

        logger.success(
            "MLflow | Modelos registrados | Dummy: {} | LR: {}",
            dummy_model_info.model_uri,
            lr_model_info.model_uri,
        )

        # 6. Persistência Legada dos Modelos e do Scaler
        # Mantemos o salvamento local em .pkl para compatibilidade com o
        # frontend Streamlit (front/app_vis.py) e o evaluate_models.py legado.
        # joblib.dump serializa o objeto Python para um arquivo binário (.pkl).
        joblib.dump(dummy, out_dir / "dummy_model.pkl")
        joblib.dump(lr, out_dir / "logistic_model.pkl")
        joblib.dump(scaler, out_dir / "scaler.pkl")

        logger.success("Artefatos salvos em '{}':", out_dir)
        logger.info("  - dummy_model.pkl")
        logger.info("  - logistic_model.pkl")
        logger.info("  - scaler.pkl")


if __name__ == "__main__":
    main()
