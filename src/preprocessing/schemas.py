"""
Módulo de Validação de Dados (Pandera)
======================================
Define o contrato de dados (schema) esperado para o dataset Telco Churn.
O uso do Pandera garante o princípio "Fail Fast", impedindo que dados
corrompidos cheguem às etapas de feature engineering e treinamento.
"""

import pandera as pa
from pandera.typing import Series


class RawDataSchema(pa.DataFrameModel):
    """
    Schema Pandera para os dados brutos (Raw Data) do Telco Churn,
    EXATAMENTE como eles saem do CSV antes de qualquer limpeza.
    """

    customerID: Series[str] = pa.Field(nullable=False)
    gender: Series[str] = pa.Field(isin=["Male", "Female"])
    SeniorCitizen: Series[int] = pa.Field(isin=[0, 1])
    Partner: Series[str] = pa.Field(isin=["Yes", "No"])
    Dependents: Series[str] = pa.Field(isin=["Yes", "No"])
    tenure: Series[int] = pa.Field(ge=0)
    PhoneService: Series[str] = pa.Field(isin=["Yes", "No"])
    MultipleLines: Series[str] = pa.Field(isin=["Yes", "No", "No phone service"])
    InternetService: Series[str] = pa.Field(isin=["DSL", "Fiber optic", "No"])
    OnlineSecurity: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    OnlineBackup: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    DeviceProtection: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    TechSupport: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    StreamingTV: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    StreamingMovies: Series[str] = pa.Field(isin=["Yes", "No", "No internet service"])
    Contract: Series[str] = pa.Field(isin=["Month-to-month", "One year", "Two year"])
    PaperlessBilling: Series[str] = pa.Field(isin=["Yes", "No"])
    PaymentMethod: Series[str] = pa.Field(
        isin=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    )
    MonthlyCharges: Series[float] = pa.Field(ge=0.0)
    
    # TotalCharges vem como string no dataset original porque tem espaços vazios (" ")
    TotalCharges: Series[str] = pa.Field(nullable=False)
    
    Churn: Series[str] = pa.Field(isin=["Yes", "No"])

    class Config:
        # strict = True proíbe a entrada de colunas não documentadas acima.
        strict = True
        # coerce = True tenta forçar automaticamente a tipagem correta se viável.
        coerce = True
