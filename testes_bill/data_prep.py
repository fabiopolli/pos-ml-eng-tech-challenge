import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    """Executa a Etapa 3.1: Tratamento e Limpeza (Data Cleaning)"""
    print(f"Carregando dados de: {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Eliminar Variáveis de Cardinalidade Inútil
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # 2. Conversão Limpa (Casting)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Preencher nulos resultantes da conversão com 0 (correspondentes a tenure = 0)
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Mapeamento do Alvo Binário (Target)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def apply_feature_engineering(df):
    """Executa a Etapa 3.5: Sugestões Avançadas de Transformação"""
    print("Aplicando Feature Engineering...")
    
    # 1. Agrupamento de tenure (Tenure Bins)
    df['Tenure_Bins'] = pd.cut(
        df['tenure'], 
        bins=[-1, 12, 24, 48, 60, 100], 
        labels=['0-12', '13-24', '25-48', '49-60', '>60']
    )
    
    # 2. Contagem de Serviços Opcionais
    services_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Services_Count'] = (df[services_cols] == 'Yes').sum(axis=1)
    
    # 3. Agrupamento Demográfico Familiar
    df['Has_Family'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
    
    # 4. Flag Unária de Tipo Pagamento
    df['Is_Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    # 5. Cálculo de Discrepância Monetária
    df['Charge_Difference'] = df['TotalCharges'] - (df['MonthlyCharges'] * df['tenure'])
    
    return df

def feature_selection_and_encoding(df):
    """
    Executa a Etapa 4: Seleção de Features Críticas 
    e a Etapa 3.2: Codificação de Variáveis
    """
    print("Realizando Seleção e Codificação de Features...")
    
    target = df['Churn']
    features = df.drop('Churn', axis=1)
    
    # Seleção estrita das features abordadas no Relatório (Seção 4) + Features de Engenharia
    critical_features = [
        'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges',
        'OnlineSecurity', 'TechSupport', 'InternetService', 'PaymentMethod',
        'Tenure_Bins', 'Services_Count', 'Has_Family', 'Is_Electronic_Check', 'Charge_Difference'
    ]
    
    features = features[critical_features]
    
    # One-Hot Encoding (Dummies) para variáveis multicasse/categóricas
    cat_cols = features.select_dtypes(include=['object', 'category']).columns
    features = pd.get_dummies(features, columns=cat_cols, drop_first=True)
    
    return features, target

def scale_features(X_train, X_test):
    """Executa a Etapa 3.3: Engenharia de Escalas de Dispersão (Z-Score)"""
    print("Aplicando Normalização (StandardScaler) nas features contínuas...")
    scaler = StandardScaler()
    
    # Vamos escalar apenas as variáveis contínuas numéricas originais e criadas
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Services_Count', 'Charge_Difference']
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train_scaled, X_test_scaled

def main():
    # Ajuste de caminhos relativos ao diretório do script
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, 'data/raw/Telco-Customer-Churn.csv')
    
    # 1. Carregamento e Limpeza
    df = load_and_clean_data(file_path)
    
    # 2. Engenharia de Features
    df = apply_feature_engineering(df)
    
    # 3. Seleção e Encoding (OHE)
    X, y = feature_selection_and_encoding(df)
    
    # 4. Split de Treino e Teste (Garantindo proporção de Churn com stratify=y)
    print("Dividindo o dataset (80% Treinamento, 20% Teste)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Scaling (Baseado em estatísticas exclusivas do treino para evitar vazamento de dados)
    X_train, X_test = scale_features(X_train, X_test)
    
    # Resumo final
    print("-" * 50)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Distribuição do Alvo no Treino:\n{y_train.value_counts(normalize=True) * 100}")
    print("-" * 50)
    
    # 6. Salvar os artefatos processados
    output_dir = os.path.join(base_path, 'data/processed')
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"Tudo finalizado! Conjuntos salvos no diretório: '{output_dir}/'")

if __name__ == "__main__":
    main()
