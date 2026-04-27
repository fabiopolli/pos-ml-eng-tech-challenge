# tests/test_schemas.py
import pandas as pd
import pytest
import pandera.errors as pa_errors

from src.preprocessing.schemas import RawDataSchema

@pytest.fixture
def valid_dataframe():
    """Fixture retornando um DataFrame de exemplo que atende ao contrato."""
    return pd.DataFrame({
        "customerID": ["7590-VHVEG", "5575-GNVDE"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [1, 34],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "No"],
        "InternetService": ["DSL", "DSL"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "No"],
        "StreamingTV": ["No", "No"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["29.85", "1889.5"],
        "Churn": ["No", "No"]
    })

def test_raw_data_schema_valid_data(valid_dataframe):
    """Garante que dados corretos passem na validação do schema sem erros."""
    # Se falhar, o Pandera lançará uma exceção
    validated_df = RawDataSchema.validate(valid_dataframe)
    assert not validated_df.empty

def test_raw_data_schema_missing_column(valid_dataframe):
    """Garante falha ao faltar uma coluna obrigatória."""
    df_missing = valid_dataframe.drop(columns=["MonthlyCharges"])
    
    with pytest.raises(pa_errors.SchemaError):
        RawDataSchema.validate(df_missing)

def test_raw_data_schema_invalid_categorical(valid_dataframe):
    """Garante falha quando um valor categórico está fora do domínio permitido."""
    df_invalid = valid_dataframe.copy()
    df_invalid.loc[0, "Contract"] = "Three years"  # Valor não existe no isin()
    
    with pytest.raises(pa_errors.SchemaError):
        RawDataSchema.validate(df_invalid)

def test_raw_data_schema_negative_numerical(valid_dataframe):
    """Garante falha quando uma variável numérica contínua viola as restrições (ex: < 0)."""
    df_invalid = valid_dataframe.copy()
    df_invalid.loc[0, "tenure"] = -5  # tenure deve ser ge=0
    
    with pytest.raises(pa_errors.SchemaError):
        RawDataSchema.validate(df_invalid)

def test_raw_data_schema_strictness(valid_dataframe):
    """Testa a regra strict=True garantindo que colunas extras não são aceitas."""
    df_extra = valid_dataframe.copy()
    df_extra["coluna_intrusa"] = "teste"
    
    with pytest.raises(pa_errors.SchemaError):
        RawDataSchema.validate(df_extra)