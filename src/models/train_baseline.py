import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# =====================================================================
# 1. CARREGAMENTO DOS DADOS E PREPARAÇÃO DO TARGET
# =====================================================================
print("Carregando os dados...")
df = pd.read_csv('data/raw/Telco-Customer-Churn.csv')

# Removendo o ID e isolando o target
X = df.drop(columns=['Churn', 'customerID']) 

# O MAPEAMENTO DO TARGET (Você faz isso aqui para garantir que vai funcionar no MLflow)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# =====================================================================
# 2. O MOCK DA FASE 2 (Até o Integrante 2 entregar a parte dele)
# =====================================================================
# Quando a fase 2 acabar, você apaga as linhas abaixo e usa:
# from src.data_engineering import preprocessor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

print("Configurando o pipeline de pré-processamento...")
# Identificando colunas numéricas e categóricas do dataset original
num_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
cat_features = [col for col in X.columns if col not in num_features]

# Um pré-processador quebra-galho só para você poder testar a Fase 3 hoje
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),
        
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ]), cat_features)
    ])

# =====================================================================
# 3. CONFIGURAÇÃO DA VALIDAÇÃO CRUZADA E MLFLOW
# =====================================================================
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}

mlflow.set_experiment("Fase3_Modelagem_Baseline")

def run_experiment(run_name, model, is_dummy=False):
    with mlflow.start_run(run_name=run_name):
        print(f"\n--- Treinando: {run_name} ---")
        
        pipeline_completo = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # O cross_validate já faz o fit e predict nos 5 folds automaticamente
        cv_results = cross_validate(pipeline_completo, X, y, cv=cv_strategy, scoring=scoring, return_estimator=True)
        
        metrics = {
            "cv_accuracy": np.mean(cv_results['test_accuracy']),
            "cv_precision": np.mean(cv_results['test_precision']),
            "cv_recall": np.mean(cv_results['test_recall']),
            "cv_f1_score": np.mean(cv_results['test_f1'])
        }
        
        if not is_dummy:
            mlflow.log_params(model.get_params())
            
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(cv_results['estimator'][0], artifact_path="model_pipeline")
        
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

# =====================================================================
# 4. EXECUTANDO OS TREINOS
# =====================================================================
print("\nIniciando experimentos...")

# Baseline 1: Dummy
run_experiment(
    run_name="DummyClassifier_Baseline", 
    model=DummyClassifier(strategy='prior'), 
    is_dummy=True
)

# Baseline 2: Regressão Logística
run_experiment(
    run_name="LogisticRegression_Baseline", 
    model=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
)

print("\n✅ Concluído! Para ver os resultados, digite 'mlflow ui' no terminal.")