import numpy as np
import pandas as pd
import pytest
from pathlib import Path


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
