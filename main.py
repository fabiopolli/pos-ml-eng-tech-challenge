"""
Orquestrador Principal do Pipeline de Treinamento
==================================================
Este é o ponto de entrada do projeto. Ao executar este script, todo o
processo de treinamento dos modelos é executado automaticamente em sequência.

O que este script faz?
----------------------
Ele é um "maestro" que coordena os dois scripts de treinamento:
  1. train_baselines.py → Treina DummyClassifier + LogisticRegression
  2. train_mlp.py       → Treina a Rede Neural MLP (PyTorch)

O design é intencionalmente simples: cada etapa é independente e poderia
ser executada isoladamente. O main.py apenas garante a ordem correta e
fornece um log visual do progresso.

Execução:
    python main.py

Próximo passo após o pipeline:
    python src/models/evaluate_models.py
"""

import time

from src.models.train_baselines import main as train_baselines
from src.models.train_mlp import main as train_mlp


def print_header(title: str) -> None:
    """
    Exibe um cabeçalho visual formatado no terminal para separar as etapas.

    Não possui lógica de negócio — é puramente cosmético para facilitar
    a leitura do log durante a execução do pipeline.

    Args:
        title: Texto a ser exibido dentro do cabeçalho.
    """
    width = 50
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:
    """
    Executa o pipeline completo de treinamento em sequência e exibe
    o tempo total de execução ao final.

    Ordem de execução:
      1. Baselines: mais rápidos, estabelecem a linha de base de performance.
      2. MLP: mais lento (requer múltiplas épocas de treino com PyTorch).

    Por que esta ordem?
    -------------------
    Os baselines são treinados primeiro pois são mais simples e rápidos.
    Se algo estiver errado com os dados ou o ambiente, descobrimos antes
    de gastar tempo treinando a rede neural.

    Nota sobre os dados:
    --------------------
    Cada etapa chama get_data_splits() internamente, que recarrega e
    transforma o CSV bruto do zero. Isso garante que cada modelo receba
    exatamente os mesmos dados, com o mesmo split, sem nenhum "vazamento"
    entre as etapas.
    """
    start_total = time.time()

    # --- ETAPA 1: Modelos de Baseline ---
    print_header("ETAPA 1/2 — Treinando Modelos Baseline")
    print("  → DummyClassifier (most_frequent)")
    print("  → LogisticRegression (class_weight='balanced')")
    print()
    start = time.time()
    train_baselines()
    print(f"  ✓ Baselines concluídos em {time.time() - start:.1f}s")

    # --- ETAPA 2: Rede Neural ---
    print_header("ETAPA 2/2 — Treinando Rede Neural (MLP)")
    print("  → ChurnMLP: Linear(input→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)")
    print()
    start = time.time()
    train_mlp()
    print(f"  ✓ MLP concluído em {time.time() - start:.1f}s")

    # --- Resumo Final ---
    print()
    print("=" * 50)
    print(f"  PIPELINE COMPLETO — {time.time() - start_total:.1f}s total")
    print("  Modelos salvos em: models/")
    print("    - dummy_model.pkl    → DummyClassifier")
    print("    - logistic_model.pkl → LogisticRegression")
    print("    - mlp_model.pth      → Rede Neural MLP (pesos PyTorch)")
    print("    - scaler.pkl         → StandardScaler (necessário para produção)")
    print("=" * 50)
    print()
    print("  Próximo passo: avalie os modelos com")
    print("  python src/models/evaluate_models.py")
    print()


if __name__ == "__main__":
    main()