import os
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from data_utils import set_seed, get_data_splits

def main():
    # 1. Configuração e Reprodutibilidade
    set_seed(42)
    
    # Definir caminhos relativos à raiz do projeto
    # Como este script está em src/, a raiz é o diretório pai
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'raw', 'Telco-Customer-Churn.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Carregamento de Dados
    print("Efetuando carga e split dos dados...")
    try:
        X_train, y_train, _, _, _, _ = get_data_splits(file_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {file_path}")
        return

    # 3. Treinamento do Modelo Dummy (Linha de base ingênua)
    print("Treinando Dummy Classifier (most_frequent)...")
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    
    # 4. Treinamento da Regressão Logística (Baseline competitiva)
    print("Treinando Regressão Logística (com balanceamento de classes)...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    # 5. Persistência dos Modelos
    dummy_path = os.path.join(models_dir, 'dummy_model.pkl')
    logistic_path = os.path.join(models_dir, 'logistic_model.pkl')
    
    joblib.dump(dummy, dummy_path)
    joblib.dump(lr, logistic_path)
    
    print("-" * 30)
    print(f"Sucesso! Modelos baselines salvos em:")
    print(f" - {dummy_path}")
    print(f" - {logistic_path}")
    print("-" * 30)

if __name__ == "__main__":
    main()
