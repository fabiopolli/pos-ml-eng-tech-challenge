"""
Script de Análise Exploratória de Dados (EDA) - Telco Customer Churn
Criado por: Data (IA) para Bill
Data: 2026-04-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(file_path):
    """Carrega o dataset e retorna um DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    df = pd.read_csv(file_path)
    return df

def basic_analysis(df):
    """Realiza análise básica do dataset (info, describe, missing values, duplicatas)."""
    print("### 2. Visualização Geral do Dataset")
    print(df.head())
    
    print("\n### 3. Informações Gerais do dataset")
    df.info()
    print("\nResumo estatístico das variáveis numéricas:")
    print(df.describe())
    
    print("\n### Análise de valores ausentes")
    missing_values = pd.DataFrame({
        'Coluna': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_values = missing_values[missing_values['Missing_Count'] > 0].sort_values(
        by='Missing_Percentage', ascending=False
    )
    
    if len(missing_values) > 0:
        print(missing_values)
    else:
        print("Nenhum missing value detectado!")

    print("\n### Verificar linhas duplicadas")
    numero_duplicatas = df.duplicated().sum()
    print(f"Número de linhas duplicadas: {numero_duplicatas}")
    if numero_duplicatas > 0:
        print(f"Percentual de duplicatas: {(numero_duplicatas / len(df) * 100):.2f}%")
    else:
        print("Nenhuma duplicata encontrada no dataset.")

def plot_target_distribution(df):
    """Analisa a distribuição da variável alvo (Churn)."""
    print("\n### 1. Análise da variável alvo (Churn)")
    churn_counts = df['Churn'].value_counts()
    churn_percent = df['Churn'].value_counts(normalize=True) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
    ax1.set_title('Contagem de Churn')
    ax1.set_xlabel('Churn')
    ax1.set_ylabel('Número de Clientes')
    
    churn_percent.plot(kind='bar', ax=ax2, color=['skyblue', 'lightcoral'])
    ax2.set_title('Percentual de Churn')
    ax2.set_xlabel('Churn')
    ax2.set_ylabel('Percentual (%)')
    
    for i, v in enumerate(churn_percent):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.show()
    print(f"Distribuição do Churn:\n{churn_percent}\n")

def plot_numeric_distributions(df):
    """Analisa as distribuições das variáveis numéricas."""
    print("\n### 2. Análise de distribuições de variáveis numéricas")
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Garantir que TotalCharges seja numérico (pode ter espaços vazios)
    df_plot = df.copy()
    df_plot['TotalCharges'] = pd.to_numeric(df_plot['TotalCharges'], errors='coerce')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(num_cols):
        sns.histplot(data=df_plot[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f'Distribuição de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequência')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    """Gera a matriz de correlação para variáveis numéricas."""
    print("\n### 3. Análise de matriz de correlações")
    df_numeric = df.copy()
    df_numeric['TotalCharges'] = pd.to_numeric(df_numeric['TotalCharges'], errors='coerce')
    
    # Selecionar apenas colunas numéricas
    corr_matrix = df_numeric.select_dtypes(include=[np.number]).corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    plt.show()

def main():
    # Raiz do projeto: notebooks/ -> projeto/
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(_base_dir, 'data', 'raw', 'Telco-Customer-Churn.csv')
    
    try:
        dataset = load_data(data_path)
        print("Dataset carregado com sucesso!")
        
        basic_analysis(dataset)
        plot_target_distribution(dataset)
        plot_numeric_distributions(dataset)
        plot_correlation_matrix(dataset)
        
    except Exception as e:
        print(f"Erro ao processar EDA: {e}")

if __name__ == "__main__":
    main()
