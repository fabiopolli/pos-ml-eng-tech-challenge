import pandas as pd

def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # remover ID
    df = df.drop(columns=["customerID"])

    # corrigir TotalCharges (string → float)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
   
    # remover nulos gerados
    df = df.dropna(subset=["TotalCharges"])

    # transformar target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df