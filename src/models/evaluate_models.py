"""
Avaliação Comparativa dos Modelos Treinados
============================================
Este script carrega os três modelos salvos e os avalia no conjunto de TESTE —
o único conjunto que nenhum modelo "viu" durante o treinamento.

Por que avaliar separadamente?
-------------------------------
Durante o treinamento, os modelos ajustam seus pesos com base nos dados de
treino e monitoram o desempenho na validação. Usar esses mesmos dados para
a avaliação final seria desonesto — seria como fazer uma prova e poder
consultar o gabarito antes. O conjunto de teste simula dados "do mundo real".

Integração com MLflow:
-----------------------
Este script cria uma run de avaliação ("evaluation_run") dentro do experimento
"churn-prediction". As métricas finais de teste (precision, recall, F1, accuracy)
de cada modelo são logadas como métricas MLflow, e o gráfico de matrizes de
confusão é logado como artefato. Isso centraliza TODOS os resultados do
pipeline na UI do MLflow para comparação.

Métricas reportadas:
  - Precisão (Precision): "De todos os que eu previ como churn, quantos realmente churnaRam?"
                          Alta precisão → poucas falsas alARmas.
  - Recall: "De todos os que realmente churnaRam, quantos eu detectei?"
            Alto recall → poucas detecções perdidas. Para retenção, esta é a
            métrica mais importante: perder um churner é mais caro que
            abordar um cliente que não iria churnar.
  - F1-Score: Média harmônica de Precisão e Recall. Bom resumo geral.
  - Acurácia: % de predições corretas. Pode ser enganosa com datasets
              desbalanceados — um modelo que nunca prevê churn teria 73%!

Artefatos gerados:
  - evaluation_summary.png → Matrizes de Confusão dos 3 modelos comparados

Execução (após rodar main.py):
    python src/models/evaluate_models.py
"""

import os
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.config import PipelineConfig
from src.models.data_utils import get_data_splits
from src.models.train_mlp import ChurnMLP


def main(
    data_path: Path | None = None,
    models_dir: Path | None = None,
    config: PipelineConfig | None = None,
) -> None:
    """
    Carrega os modelos treinados, gera predições no conjunto de teste e
    registra métricas e artefatos no MLflow.

    Estratégia de carregamento dos modelos:
    ----------------------------------------
    1. Tenta carregar via MLflow Model Registry (fonte canônica de verdade).
    2. Se o modelo não estiver no Registry (ex: primeira execução sem MLflow,
       ou register_model=False), faz fallback para os arquivos .pkl/.pth locais.

    Args:
        data_path: Caminho para o CSV bruto. Se None, usa o padrão do projeto.
        models_dir: Diretório com os modelos salvos. Se None, usa 'models/'.
        config: Configuração do pipeline. Se None, usa PipelineConfig() com padrões.
    """
    cfg = config or PipelineConfig()

    BASE_DIR = Path(__file__).resolve().parents[2]
    file_path = data_path or (BASE_DIR / "data" / "raw" / "Telco-Customer-Churn.csv")
    out_dir = models_dir or (BASE_DIR / "models")

    # 1. Carregamento do Conjunto de Teste
    # Usamos APENAS X_test e y_test (5º e 6º elementos da tupla).
    # Os demais são ignorados com "_" pois já foram usados no treinamento.
    # É fundamental usar get_data_splits com os mesmos parâmetros do treino
    # para garantir que os dados passem pelas MESMAS transformações.
    logger.info("Carregando dados de teste (final evaluation)...")
    try:
        _, _, _, _, X_test, y_test, _ = get_data_splits(file_path)
    except FileNotFoundError:
        logger.error("Dataset não encontrado em {}", file_path)
        return

    # 2. Carregamento dos Modelos
    # Tentamos primeiro o MLflow Registry; se falhar, usamos os arquivos locais.
    logger.info("Carregando modelos persistidos...")
    try:
        dummy = _load_sklearn_model("ChurnDummyClassifier", out_dir / "dummy_model.pkl")
        lr = _load_sklearn_model("ChurnLogisticRegression", out_dir / "logistic_model.pkl")
        mlp = _load_pytorch_model(out_dir / "mlp_model.pth", X_test.shape[1])
    except FileNotFoundError as e:
        logger.error("Um ou mais modelos não encontrados: {}", e)
        logger.info("Execute 'python main.py' para treinar os modelos primeiro.")
        return

    # 3. Geração de Predições
    # Cada modelo gera suas predições de forma diferente:
    logger.info("Gerando predições nos dados de teste...")

    # sklearn: .predict() retorna diretamente 0 ou 1.
    y_pred_dummy = dummy.predict(X_test)
    y_prob_dummy = dummy.predict_proba(X_test)[:, 1]
    
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    # PyTorch: converte o DataFrame para Tensor, passa pela rede (obtém logits),
    # aplica sigmoid (converte logit em probabilidade 0-1) e decide pelo limiar 0.5.
    # Probabilidade > 0.5 → prevê churn (1). Abaixo → não churn (0).
    X_test_t = torch.tensor(X_test.astype(np.float32).values, dtype=torch.float32)
    with torch.no_grad():  # Sem gradientes — economia de memória na avaliação
        logits = mlp(X_test_t)
        # sigmoid converte o logit (qualquer número real) para [0, 1]
        probs_mlp = torch.sigmoid(logits)
        y_prob_mlp = probs_mlp.numpy().flatten()
        # .int() converte True/False para 1/0
        # .numpy().flatten() converte para array 1D do NumPy (formato sklearn)
        y_pred_mlp = (probs_mlp > 0.5).int().numpy().flatten()

    # 4. Relatório de Métricas no Terminal
    # classification_report mostra Precisão, Recall, F1 e suporte (N amostras)
    # para cada classe (0 = não churn, 1 = churn) e as médias globais.
    print("\n" + "=" * 60)
    print("            RELATÓRIO DE AVALIAÇÃO COMPARATIVA")
    print("=" * 60)
    print("\n[1] MODELO DUMMY (Baseline Ingênua)")
    print(classification_report(y_test, y_pred_dummy, zero_division=0))
    print("\n[2] REGRESSÃO LOGÍSTICA (Balanced)")
    print(classification_report(y_test, y_pred_lr))
    print("\n[3] REDE NEURAL MLP (Deep Learning)")
    print(classification_report(y_test, y_pred_mlp))

    # 5. Visualização das Matrizes de Confusão
    # A Matriz de Confusão mostra, em formato de grade:
    #   - Verdadeiros Negativos (TN): previu 0, era 0 → Acerto (não churn)
    #   - Falsos Positivos (FP):      previu 1, era 0 → Alarme falso
    #   - Falsos Negativos (FN):      previu 0, era 1 → Churner não detectado (pior caso)
    #   - Verdadeiros Positivos (TP): previu 1, era 1 → Acerto (churn detectado)
    logger.info("Gerando visualização das Matrizes de Confusão...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_configs = [
        ("Dummy Classifier", y_pred_dummy),
        ("Logistic Regression", y_pred_lr),
        ("Neural Network (MLP)", y_pred_mlp),
    ]

    for i, (name, y_pred) in enumerate(plot_configs):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            ax=axes[i], cbar=False,
        )
        axes[i].set_title(name)
        axes[i].set_xlabel("Predito (Classe)")
        axes[i].set_ylabel("Real (Classe)")

    plt.suptitle(
        "Comparativo de Matrizes de Confusão no Conjunto de Teste",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Salva o gráfico como imagem PNG na raiz do projeto.
    # O app Streamlit (front/app_vis.py) também exibe esta imagem.
    output_plot = BASE_DIR / "evaluation_summary.png"
    plt.savefig(output_plot)
    logger.success("Gráfico comparativo salvo em: {}", output_plot)
    plt.close()  # Libera memória — importante em ambientes sem display gráfico

    # =========================================================================
    # 6. Log das Métricas e Artefatos no MLflow
    # =========================================================================
    # Configuramos o experimento MLflow para que a run de avaliação fique
    # agrupada com as runs de treinamento na UI do MLflow.
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
 
    with mlflow.start_run(run_name="evaluation_run", nested=True):
        # --- Log de Métricas Finais de Teste ---
        # Estas são as métricas "honestas" — calculadas no conjunto de teste,
        # que NENHUM modelo viu durante o treinamento. São os números que
        # definem a qualidade real do modelo em produção.
        # O prefixo (dummy_, lr_, mlp_) identifica qual modelo gerou a métrica.
        _log_test_metrics(y_test, y_pred_dummy, y_prob_dummy, prefix="dummy")
        _log_test_metrics(y_test, y_pred_lr, y_prob_lr, prefix="lr")
        _log_test_metrics(y_test, y_pred_mlp, y_prob_mlp, prefix="mlp")

        # --- Log do Artefato Visual ---
        # mlflow.log_artifact salva o arquivo como artefato da run, tornando-o
        # acessível diretamente na UI do MLflow (aba "Artifacts" da run).
        mlflow.log_artifact(str(output_plot))
        logger.success(
            "MLflow | Métricas de avaliação e artefato 'evaluation_summary.png' logados."
        )


def _load_sklearn_model(registry_name: str, local_path: Path):
    """
    Tenta carregar um modelo sklearn do MLflow Model Registry.
    Em caso de falha (modelo não registrado), faz fallback para o arquivo local.

    A URI "models:/NomeDoModelo/latest" instrui o MLflow a buscar a versão
    mais recente do modelo registrado, independentemente do número da versão.

    Args:
        registry_name: Nome do modelo no MLflow Model Registry.
        local_path: Caminho do arquivo .pkl como fallback local.

    Returns:
        O modelo sklearn carregado e pronto para predição.
    """
    try:
        model = mlflow.sklearn.load_model(f"models:/{registry_name}/latest")
        logger.info("MLflow Registry | Modelo '{}' carregado.", registry_name)
        return model
    except Exception:
        logger.warning(
            "Modelo '{}' não encontrado no Registry. Carregando de '{}'.",
            registry_name, local_path,
        )
        return joblib.load(local_path)


def _load_pytorch_model(local_path: Path, input_dim: int) -> ChurnMLP:
    """
    Carrega o modelo PyTorch (MLP) do arquivo .pth local.

    Para a MLP, usamos o carregamento local direto pois o carregamento
    via mlflow.pytorch.load_model retorna um wrapper que pode diferir
    da interface nativa do ChurnMLP (necessária para o evaluate_models).

    Args:
        local_path: Caminho do arquivo .pth com os pesos do modelo.
        input_dim: Número de features de entrada (deve bater com o treino).

    Returns:
        A instância de ChurnMLP com os pesos carregados, em modo eval().
    """
    mlp = ChurnMLP(input_dim=input_dim)
    mlp.load_state_dict(torch.load(local_path, weights_only=True))
    # model.eval() desativa Dropout para avaliação estável e determinística.
    mlp.eval()
    logger.info("Modelo MLP carregado de '{}'.", local_path)
    return mlp


def _log_test_metrics(y_true, y_pred, y_prob, prefix: str) -> None:
    """
    Calcula e loga as principais métricas de classificação no MLflow.

    As métricas são nomeadas com um prefixo para identificar o modelo:
      "{prefix}_test_accuracy", "{prefix}_test_precision", etc.

    Por que usar 'weighted' nas médias?
    ------------------------------------
    Com datasets desbalanceados (como o de churn, onde ~73% não churnam),
    a média ponderada (weighted) é mais representativa que a macro média,
    pois dá mais peso às classes com mais amostras.

    Args:
        y_true: Rótulos reais (ground truth).
        y_pred: Predições do modelo.
        y_prob: Probabilidades preditas (para cálculo de AUC-ROC).
        prefix: Prefixo para nomear as métricas (ex: "dummy", "lr", "mlp").
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Premissas Financeiras Fixas para o MLflow
    ticket_medio = 70.0
    custo_retencao = 30.0

    # Custo sem modelo: Perdemos a receita de todos os clientes que cancelariam (TP + FN)
    custo_sem_modelo = (tp + fn) * ticket_medio

    # Custo com modelo: 
    # - Gastamos a retenção com quem o modelo alertou (TP e FP)
    # - Perdemos a receita de quem o modelo deixou passar (FN)
    # - Assumimos retenção total dos TPs devido à ação.
    custo_com_modelo = ((tp + fp) * custo_retencao) + (fn * ticket_medio)

    economia_estimada = custo_sem_modelo - custo_com_modelo

    mlflow.log_metrics({
        f"{prefix}_test_accuracy":  accuracy_score(y_true, y_pred),
        f"{prefix}_test_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_test_recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_test_f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_test_auc_roc":   roc_auc_score(y_true, y_prob),
        f"{prefix}_test_estimated_savings": economia_estimada,
    })


if __name__ == "__main__":
    main()
