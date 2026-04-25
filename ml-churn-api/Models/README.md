# Modelos ML

Pasta para armazenar os arquivos de modelo de machine learning.

## 📦 Arquivos de Modelo

| Arquivo | Descrição |
|---------|-----------|
| `churn_model.pkl` | Modelo de predição de churn (joblib/pickle) |

---

## 🚀 Como adicionar um modelo

### 1. Treinar o modelo

```python
# training/train_model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

# Exemplo com dados fictícios
# Substitua pelo seu dataset real
data = pd.read_csv("seu_dataset.csv")

X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Criar pipeline com scaler + modelo
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100))
])

# Treinar
pipeline.fit(X_train, y_train)

# Avaliar
score = pipeline.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")

# Salvar
joblib.dump(pipeline, "churn_model.pkl")
```

### 2. Colocar o arquivo

Salve o arquivo do modelo nesta pasta:
```
ml-churn-api/models/churn_model.pkl
```

### 3. Atualizar a API para usar o modelo real

Edite `app/services/model_service.py`:

```python
from app.models import get_model, prepare_features

def predict(data: PredictInput) -> PredictOutput:
    # Carregar modelo real
    model = get_model("churn_model.pkl")
    
    if not model.is_loaded:
        model.load()
    
    # Preparar features
    features = prepare_features(data.dict())
    
    # Predição
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]
    
    return PredictOutput(...)
```

---

## 📋 Formatos suportados

| Formato | Extensão | Biblioteca |
|---------|----------|------------|
| **joblib** | `.pkl`, `.joblib` | scikit-learn |
| **pickle** | `.pickle` | Python padrão |
| **ONNX** | `.onnx` | onnxruntime |
| **TensorFlow** | `.h5`, `.pb` | tensorflow |
| **PyTorch** | `.pt`, `.pth` | torch |

---

## 📊 Exemplo de dataset para churn

```csv
tenure,monthly_charges,total_charges,contract_type,payment_method,has_phone_service,has_internet_service,has_online_security,has_online_backup,has_device_protection,has_tech_support,streaming_tv,streaming_movies,churn
12,70.0,840.0,one_year,credit_card,1,1,0,1,0,0,0,0,0
24,80.0,1920.0,two_year,credit_card,1,1,1,1,1,1,1,1,0
3,50.0,150.0,monthly,electronic_check,1,1,0,0,0,0,0,0,1
```

---

## 🔧 Scripts de treinamento

Coloque scripts de treinamento na pasta `training/`:

```
ml-churn-api/
├── models/
│   └── churn_model.pkl    ← Modelo treinado
├── training/
│   ├── train_model.py     ← Script de treino
│   └── evaluate.py       ← Avaliação
└── notebooks/
    └── eda.ipynb          ← Análise exploratória
```