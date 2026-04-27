import os
import mlflow
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def mlflow_test_tracking(tmp_path: Path):
    """
    Fixture de isolamento do MLflow para todos os testes.

    Por que isso é necessário?
    --------------------------
    As funções de treinamento (train_baselines, train_mlp) agora chamam
    mlflow.start_run() internamente. Sem este isolamento, cada execução
    dos testes criaria runs reais no diretório 'mlruns/' do projeto,
    poluindo os experimentos de desenvolvimento com dados sintéticos.

    Esta fixture usa `autouse=True` para ser aplicada AUTOMATICAMENTE a
    todos os testes sem precisar declará-la explicitamente.

    O que ela faz:
    1. Redireciona o tracking URI para um diretório temporário único
       por sessão de teste (é deletado automaticamente pelo pytest).
    2. Desabilita o registro no Model Registry (register_model=False)
       para evitar chamadas ao registry durante os testes.
    3. Restaura as configurações originais após cada teste (via yield).
    """
    test_mlruns = tmp_path / "mlruns"
    
    # Identifica o sistema operacional para evitar o KeyError de URI no MLflow
    if os.name == "nt":
        # No Windows, força o formato URI (file:///C:/...)
        tracking_uri = test_mlruns.resolve().as_uri()
    else:
        # No Linux/Mac, o caminho em string padrão funciona perfeitamente
        tracking_uri = str(test_mlruns)

    mlflow.set_tracking_uri(tracking_uri)
    
    yield
    
    # Restaura o tracking URI padrão após o teste
    mlflow.set_tracking_uri("mlruns")


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """
    Cria um CSV sintético com a estrutura completa do dataset Telco Churn.
    Usa n=200 para garantir splits estratificados estáveis (70/15/15).
    """
    rng = np.random.default_rng(42)
    n = 200

    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 120, n).round(2)

    # TotalCharges em branco para clientes com tenure=0 (replica o dataset real)
    total_charges = [
        " " if t == 0 else str(round(float(m) * int(t), 2))
        for t, m in zip(tenure, monthly)
    ]

    data = {
        "customerID": [f"ID-{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            n,
        ),
        "MonthlyCharges": monthly,
        "TotalCharges": total_charges,
        "Churn": rng.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    }

    csv_path = tmp_path / "test_churn.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path
