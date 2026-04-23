"""
Pipeline Completo de Treinamento - Telco Churn Prediction
=========================================================
Orquestra as etapas do pipeline ML em sequência:
  1. Baselines (DummyClassifier + LogisticRegression)
  2. Rede Neural (MLP com PyTorch)

Execute a partir da raiz do projeto:
    python main.py
"""

import sys
import os
import time

# Garante que src/models/ esteja acessível para os imports internos dos módulos
_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'models')
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from src.models.train_baselines import main as train_baselines
from src.models.train_mlp import main as train_mlp


def print_header(title: str) -> None:
    """Exibe um cabeçalho formatado para separar as etapas do pipeline."""
    width = 50
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main():
    start_total = time.time()

    print_header("ETAPA 1/2 — Treinando Modelos Baseline")
    print("  → DummyClassifier (most_frequent)")
    print("  → LogisticRegression (class_weight='balanced')")
    print()
    start = time.time()
    train_baselines()
    print(f"  ✓ Baselines concluídos em {time.time() - start:.1f}s")

    print_header("ETAPA 2/2 — Treinando Rede Neural (MLP)")
    print("  → ChurnMLP: Linear(input→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)")
    print()
    start = time.time()
    train_mlp()
    print(f"  ✓ MLP concluído em {time.time() - start:.1f}s")

    print()
    print("=" * 50)
    print(f"  PIPELINE COMPLETO — {time.time() - start_total:.1f}s total")
    print("  Modelos salvos em: models/")
    print("    - dummy_model.pkl")
    print("    - logistic_model.pkl")
    print("    - mlp_model.pth")
    print("=" * 50)
    print()
    print("  Próximo passo: avalie os modelos com")
    print("  python src/models/evaluate_models.py")
    print()


if __name__ == "__main__":
    main()