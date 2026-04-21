import pandas as pd
from src.preprocessing import limpar_dados, criar_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# carregar dados
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# limpeza
df = limpar_dados(df)

# separar X e y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# pipeline
pipeline = criar_pipeline()

# aplicar transformações
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

#  Validacao 

print("Shape original:", X.shape)
print("Shape treino:", X_train_processed.shape)
print("Shape teste:", X_test_processed.shape)

print("Tipo do dado:", type(X_train_processed))

print("Tem NaN?", np.isnan(X_train_processed).any())

#  TESTE  MODELO
model = LogisticRegression(max_iter=1000)
model.fit(X_train_processed, y_train)

score = model.score(X_test_processed, y_test)

print("Acurácia do modelo:", score)

print("Pipeline executado com sucesso ")