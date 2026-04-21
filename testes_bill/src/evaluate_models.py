import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data_utils import get_data_splits
from train_mlp import ChurnMLP

def main():
    # Caminhos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'raw', 'Telco-Customer-Churn.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    # 1. Carregamento do Conjunto de Teste (Inédito)
    print("Carregando dados de teste (final evaluation)...")
    try:
        _, _, _, _, X_test, y_test = get_data_splits(file_path)
    except FileNotFoundError:
        print("Erro: Dataset não encontrado.")
        return

    # 2. Carregamento dos Modelos Salvos
    print("Carregando modelos persistidos em /models/...")
    try:
        dummy = joblib.load(os.path.join(models_dir, 'dummy_model.pkl'))
        lr = joblib.load(os.path.join(models_dir, 'logistic_model.pkl'))
        
        # Recriar arquitetura e carregar pesos da MLP
        mlp = ChurnMLP(X_test.shape[1])
        mlp.load_state_dict(torch.load(os.path.join(models_dir, 'mlp_model.pth')))
        mlp.eval() # Modo de avaliação
    except FileNotFoundError as e:
        print(f"Erro: Um ou mais modelos não encontrados. {e}")
        return

    # 3. Geração de Predições
    print("Gerando predições nos dados de teste...")
    
    # Baselines
    y_pred_dummy = dummy.predict(X_test)
    y_pred_lr = lr.predict(X_test)
    
    # Rede Neural
    X_test_t = torch.tensor(X_test.astype(np.float32).values, dtype=torch.float32)
    with torch.no_grad():
        logits = mlp(X_test_t)
        probs = torch.sigmoid(logits)
        y_pred_mlp = (probs > 0.5).int().numpy().flatten()
        
    # 4. Relatório de Métricas Numéricas
    print("\n" + "="*60)
    print("            RELATÓRIO DE AVALIAÇÃO COMPARATIVA")
    print("="*60)
    
    print("\n[1] MODELO DUMMY (Baseline Ingênua)")
    print(classification_report(y_test, y_pred_dummy, zero_division=0))
    
    print("\n[2] REGRESSÃO LOGÍSTICA (Balanced)")
    print(classification_report(y_test, y_pred_lr))
    
    print("\n[3] REDE NEURAL MLP (Deep Learning)")
    print(classification_report(y_test, y_pred_mlp))
    
    # 5. Visualização: Matrizes de Confusão Comparativas
    print("Gerando visualização das Matrizes de Confusão...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    configs = [
        ('Dummy Classifier', y_pred_dummy),
        ('Logistic Regression', y_pred_lr),
        ('Neural Network (MLP)', y_pred_mlp)
    ]
    
    for i, (name, y_pred) in enumerate(configs):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name}')
        axes[i].set_xlabel('Predito (Classe)')
        axes[i].set_ylabel('Real (Classe)')
        
    plt.suptitle('Comparativo de Matrizes de Confusão no Conjunto de Teste', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salvar a imagem do gráfico
    output_plot = os.path.join(base_dir, 'evaluation_summary.png')
    plt.savefig(output_plot)
    print(f"\nGráfico comparativo salvo em: {output_plot}")
    
    # Nota: Em ambientes sem display, plt.show() pode ser ignorado ou causar erro.
    # Como salvamos o arquivo, o objetivo principal está cumprido.
    plt.close()

if __name__ == "__main__":
    main()
