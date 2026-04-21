import pandas as pd
import numpy as np
import os
import random
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    """Garante reprodutibilidade em todas as bibliotecas."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Configurações adicionais para determinismo no PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_data_splits(file_path):
    """
    Lê o dataset bruto, aplica limpeza, engenharia de features,
    encoding e divide em Treino (70%), Validação (15%) e Teste (15%).
    Retorna (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    # 1. Carga e Limpeza Básica
    df = pd.read_csv(file_path)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # Conversão de TotalCharges para numérico
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Mapeamento do Target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # 2. Feature Engineering
    # Agrupamento de tenure
    df['Tenure_Bins'] = pd.cut(
        df['tenure'], 
        bins=[-1, 12, 24, 48, 60, 100], 
        labels=['0-12', '13-24', '25-48', '49-60', '>60']
    )
    
    # Contagem de serviços
    services_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Services_Count'] = (df[services_cols] == 'Yes').sum(axis=1)
    
    # Flags binárias
    df['Has_Family'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
    df['Is_Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    # Diferença de cobrança
    df['Charge_Difference'] = df['TotalCharges'] - (df['MonthlyCharges'] * df['tenure'])

    # 3. Seleção e Encoding
    target = df['Churn']
    critical_features = [
        'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges',
        'OnlineSecurity', 'TechSupport', 'InternetService', 'PaymentMethod',
        'Tenure_Bins', 'Services_Count', 'Has_Family', 'Is_Electronic_Check', 'Charge_Difference'
    ]
    features = df[critical_features]
    
    # One-Hot Encoding
    cat_cols = features.select_dtypes(include=['object', 'category']).columns
    features = pd.get_dummies(features, columns=cat_cols, drop_first=True, dtype=int)

    # 4. Divisão dos Dados (Splits)
    # 70% Treino, 30% Temporário (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, target, test_size=0.3, random_state=42, stratify=target
    )

    # Divide o Temporário em 50/50 (Resultando em 15% cada do total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 5. Padronização (Scaling)
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Services_Count', 'Charge_Difference']
    
    # Ajuste e transformação no Treino
    X_train = X_train.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    
    # Transformação em Validação e Teste (usando estatísticas do Treino)
    X_val = X_val.copy()
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    
    X_test = X_test.copy()
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, y_train, X_val, y_val, X_test, y_test
