import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocessing.data_prep import (
    apply_feature_engineering,
    feature_selection_and_encoding,
    load_and_clean_data,
    scale_and_split,
)


def test_load_remove_customer_id(sample_csv_path):
    df = load_and_clean_data(sample_csv_path)
    assert "customerID" not in df.columns


def test_load_total_charges_numerico(sample_csv_path):
    df = load_and_clean_data(sample_csv_path)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0


def test_load_churn_binario(sample_csv_path):
    df = load_and_clean_data(sample_csv_path)
    assert set(df["Churn"].dropna().unique()).issubset({0, 1})


def test_feature_engineering_cria_colunas(sample_csv_path):
    df = load_and_clean_data(sample_csv_path)
    df = apply_feature_engineering(df)
    novas = ["Tenure_Bins", "Services_Count", "Has_Family", "Is_Electronic_Check", "Charge_Difference"]
    for col in novas:
        assert col in df.columns, f"Coluna esperada '{col}' não encontrada"


def test_encoding_sem_colunas_categoricas(sample_csv_path):
    df = load_and_clean_data(sample_csv_path)
    df = apply_feature_engineering(df)
    X, _ = feature_selection_and_encoding(df)
    cat_restantes = X.select_dtypes(include=["object", "category"]).columns.tolist()
    assert cat_restantes == [], f"Ainda há colunas categóricas: {cat_restantes}"


def test_encoding_dtype_int(sample_csv_path):
    df = load_and_clean_data(sample_csv_path)
    df = apply_feature_engineering(df)
    X, _ = feature_selection_and_encoding(df)
    # Colunas binárias OHE devem conter apenas 0 e 1 (dtype int)
    # Identifica colunas binárias excluindo as numéricas contínuas conhecidas
    numericas_continuas = {"tenure", "MonthlyCharges", "TotalCharges", "Services_Count", "Charge_Difference"}
    dummy_cols = [c for c in X.columns if c not in numericas_continuas]
    for col in dummy_cols:
        valores_unicos = set(X[col].unique())
        assert valores_unicos.issubset({0, 1}), f"Coluna OHE '{col}' contém valores além de 0/1: {valores_unicos}"


def test_scale_and_split_proporcoes(sample_csv_path):
    df = load_and_clean_data(sample_csv_path)
    df = apply_feature_engineering(df)
    X, y = feature_selection_and_encoding(df)
    X_train, y_train, X_val, y_val, X_test, y_test, _ = scale_and_split(X, y)

    total = len(X_train) + len(X_val) + len(X_test)
    assert abs(len(X_train) / total - 0.70) < 0.05, "Proporção de treino fora do esperado"
    assert abs(len(X_val) / total - 0.15) < 0.05, "Proporção de validação fora do esperado"
    assert abs(len(X_test) / total - 0.15) < 0.05, "Proporção de teste fora do esperado"


def test_scale_and_split_retorna_7_elementos(sample_csv_path):
    df = load_and_clean_data(sample_csv_path)
    df = apply_feature_engineering(df)
    X, y = feature_selection_and_encoding(df)
    result = scale_and_split(X, y)
    assert len(result) == 7
    assert isinstance(result[-1], StandardScaler)


def test_scale_and_split_sem_data_leakage(sample_csv_path):
    """Scaler deve ser ajustado só no treino: média do treino ≈ 0, val/test não."""
    df = load_and_clean_data(sample_csv_path)
    df = apply_feature_engineering(df)
    X, y = feature_selection_and_encoding(df)
    X_train, _, X_val, _, X_test, _, _ = scale_and_split(X, y)

    # Treino deve ter média ~0 em colunas numéricas escaladas
    assert abs(X_train["tenure"].mean()) < 0.5
    # Val e Test devem ter médias diferentes do treino
    assert X_val["tenure"].mean() != 0.0 or X_test["tenure"].mean() != 0.0
