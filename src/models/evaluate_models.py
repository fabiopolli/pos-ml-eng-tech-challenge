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

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from src.models.data_utils import get_data_splits
from src.models.train_mlp import ChurnMLP


def main(
    data_path: Path | None = None,
    models_dir: Path | None = None,
) -> None:
    """
    Carrega os modelos treinados, gera predições no conjunto de teste e
    exibe métricas e visualizações comparativas.

    Args:
        data_path: Caminho para o CSV bruto. Se None, usa o padrão do projeto.
        models_dir: Diretório com os modelos salvos. Se None, usa 'models/'.
    """
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

    # 2. Carregamento dos Modelos Salvos
    # joblib.load desserializa os objetos do disco. Os modelos sklearn
    # ficam prontos para uso imediatamente após o carregamento.
    logger.info("Carregando modelos persistidos em '{}'...", out_dir)
    try:
        dummy = joblib.load(out_dir / "dummy_model.pkl")
        lr = joblib.load(out_dir / "logistic_model.pkl")

        # Para a MLP, precisamos: (a) recriar a arquitetura com os mesmos
        # parâmetros usados no treino, e (b) carregar os pesos salvos.
        # Se hidden_dims ou input_dim não baterem, load_state_dict vai falhar.
        mlp = ChurnMLP(input_dim=X_test.shape[1])
        mlp.load_state_dict(torch.load(out_dir / "mlp_model.pth"))

        # model.eval() desativa Dropout para avaliação estável e determinística.
        mlp.eval()
    except FileNotFoundError as e:
        logger.error("Um ou mais modelos não encontrados: {}", e)
        logger.info("Execute 'python main.py' para treinar os modelos primeiro.")
        return

    # 3. Geração de Predições
    # Cada modelo gera suas predições de forma diferente:
    logger.info("Gerando predições nos dados de teste...")

    # sklearn: .predict() retorna diretamente 0 ou 1.
    y_pred_dummy = dummy.predict(X_test)
    y_pred_lr = lr.predict(X_test)

    # PyTorch: converte o DataFrame para Tensor, passa pela rede (obtém logits),
    # aplica sigmoid (converte logit em probabilidade 0-1) e decide pelo limiar 0.5.
    # Probabilidade > 0.5 → prevê churn (1). Abaixo → não churn (0).
    X_test_t = torch.tensor(X_test.astype(np.float32).values, dtype=torch.float32)
    with torch.no_grad():  # Sem gradientes — economia de memória na avaliação
        logits = mlp(X_test_t)
        # sigmoid converte o logit (qualquer número real) para [0, 1]
        # .int() converte True/False para 1/0
        # .numpy().flatten() converte para array 1D do NumPy (formato sklearn)
        y_pred_mlp = (torch.sigmoid(logits) > 0.5).int().numpy().flatten()

    # 4. Relatório de Métricas
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

    configs = [
        ("Dummy Classifier", y_pred_dummy),
        ("Logistic Regression", y_pred_lr),
        ("Neural Network (MLP)", y_pred_mlp),
    ]

    for i, (name, y_pred) in enumerate(configs):
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


if __name__ == "__main__":
    main()
