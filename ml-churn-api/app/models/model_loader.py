"""
Módulo de carregamento e inferência de modelos ML.

Este arquivo contém funções utilitárias para carregar e usar
modelos de machine learning na API.

Suporta:
- Modelos .pkl (sklearn/joblib)
- Modelos .pth (PyTorch) via wrapper
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parents[3]  # raiz do projeto
MODELS_DIR = BASE_DIR / "models"


class ModelLoader:
    """Classe para gerenciar o carregamento de modelos ML."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
    
    def load(self) -> bool:
        """
        Carrega o modelo do arquivo.
        
        Returns:
            bool: True se carregou com sucesso
        """
        try:
            full_path = os.path.join(MODELS_DIR, self.model_path)
            
            if not os.path.exists(full_path):
                logger.warning(f"Modelo não encontrado: {full_path}")
                return False
            
            self.model = joblib.load(full_path)
            self.is_loaded = True
            logger.info(f"Modelo carregado: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def predict(self, data: Any) -> Any:
        """
        Executa predição com o modelo.
        
        Args:
            data: Dados de entrada (DataFrame, array, etc.)
            
        Returns:
            Resultado da predição
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado")
        
        return self.model.predict(data)
    
    def predict_proba(self, data: Any) -> Any:
        """
        Executa predição com probabilidades.
        
        Args:
            data: Dados de entrada
            
        Returns:
            Array de probabilidades
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado")
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(data)
        raise AttributeError("Modelo não suporta predict_proba")


# Instância global do modelo
_model_instance: Optional[ModelLoader] = None


def get_model(model_path: str = "churn_model.pkl") -> ModelLoader:
    """
    Obtém instância singleton do modelo.
    
    Args:
        model_path: Caminho do arquivo do modelo
        
    Returns:
        ModelLoader: Instância do modelo
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = ModelLoader(model_path)
    
    return _model_instance


def load_model(model_path: str = "churn_model.pkl") -> bool:
    """
    Função de conveniência para carregar o modelo.
    
    Args:
        model_path: Caminho do arquivo do modelo
        
    Returns:
        bool: True se carregou com sucesso
    """
    model = get_model(model_path)
    return model.load()


def auto_load_model() -> bool:
    """
    Detecta e carrega o modelo disponível automaticamente.
    
    Prioridade:
    1. churn_model.pkl (modelo sklearn/joblib)
    2. mlp_model.pkl (wrapper PyTorch)
    3. mlp_model.pth (modelo PyTorch nativo)
    
    Returns:
        bool: True se carregou com sucesso
    """
    # Tenta carregar modelo .pkl primeiro
    pkl_path = MODELS_DIR / "churn_model.pkl"
    if pkl_path.exists():
        logger.info(f"Carregando modelo: {pkl_path}")
        return load_model("churn_model.pkl")
    
    # Tenta wrapper PyTorch
    mlp_pkl = MODELS_DIR / "mlp_model.pkl"
    if mlp_pkl.exists():
        logger.info(f"Carregando wrapper PyTorch: {mlp_pkl}")
        return load_model("mlp_model.pkl")
    
    # Tenta modelo PyTorch nativo
    pth_path = MODELS_DIR / "mlp_model.pth"
    if pth_path.exists():
        logger.info(f"Modelo PyTorch detectado: {pth_path}")
        logger.info("Execute: python -m app.models.pytorch_wrapper para converter")
        return False
    
    logger.warning("Nenhum modelo encontrado")
    return False


# ============================================================
# EXEMPLO: Como preparar seus dados para o modelo
# ============================================================

def prepare_features(data: dict) -> list:
    """
    Prepara features para inferência.
    
    Este é um exemplo de como transformar os dados de entrada
    no formato esperado pelo modelo.
    
    Args:
        data: Dicionário com dados do cliente
        
    Returns:
        list: Features no formato do modelo
    """
    # Mapeamento de contract_type
    contract_map = {
        "month": [1, 0, 0],
        "one_year": [0, 1, 0],
        "two_year": [0, 0, 1]
    }
    
    # Mapeamento de payment_method
    payment_map = {
        "credit_card": [1, 0, 0, 0],
        "debit_card": [0, 1, 0, 0],
        "electronic_check": [0, 0, 1, 0],
        "bank_transfer": [0, 0, 0, 1]
    }
    
    features = [
        data.get("tenure", 0),
        data.get("monthly_charges", 0),
        data.get("total_charges", 0),
    ]
    
    # Adicionar one-hot encoding de contract_type
    features.extend(contract_map.get(data.get("contract_type", "month"), [1, 0, 0]))
    
    # Adicionar one-hot encoding de payment_method
    features.extend(payment_map.get(data.get("payment_method", "credit_card"), [1, 0, 0, 0]))
    
    # Features booleanas
    features.extend([
        int(data.get("has_phone_service", False)),
        int(data.get("has_internet_service", False)),
        int(data.get("has_online_security", False)),
        int(data.get("has_online_backup", False)),
        int(data.get("has_device_protection", False)),
        int(data.get("has_tech_support", False)),
        int(data.get("streaming_tv", False)),
        int(data.get("streaming_movies", False)),
    ])
    
    return features


# ============================================================
# INSTRUÇÕES PARA ADICIONAR UM MODELO REAL
# ============================================================

"""
PASSOS PARA ADICIONAR UM MODELO REAL:

1. TREINAR O MODELO
   - Use scikit-learn, XGBoost, LightGBM, etc.
   - Exemplo:
   
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   
   # Dados de exemplo
   X, y = load_seu_dataset()
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   
   # Treinar
   model = RandomForestClassifier(n_estimators=100)
   model.fit(X_train, y_train)
   
   # Salvar
   joblib.dump(model, 'models/churn_model.pkl')


2. COLOCAR O ARQUIVO DO MODELO
   - Salve o modelo em: ml-churn-api/models/churn_model.pkl
   - Formatos suportados: .pkl, .joblib, .pickle


3. ATUALIZAR O model_service.py
   - Modifique a função _calculate_churn_score()
   - Use o modelo real para inferência:
   
   from app.models.model_loader import get_model
   
   def predict(data: PredictInput) -> PredictOutput:
       model = get_model("churn_model.pkl")
       
       if not model.is_loaded:
           model.load()
       
       features = prepare_features(data.dict())
       prediction = model.predict([features])
       
       ...


4. FORMATOS DE MODELO SUPORTADOS
   - joblib (.pkl, .joblib)
   - pickle (.pickle)
   - ONNX (onnxruntime)
   - TensorFlow (tf.keras)
   - PyTorch (.pt)


5. EXEMPLO DE pipeline COMPLETO

   # training/train_model.py
   import joblib
   from sklearn.ensemble import GradientBoostingClassifier
   from sklearn.preprocessing import StandardScaler
   from sklearn.pipeline import Pipeline
   
   # Criar pipeline com scaler + modelo
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('classifier', GradientBoostingClassifier())
   ])
   
   # Treinar
   pipeline.fit(X_train, y_train)
   
   # Salvar
   joblib.dump(pipeline, 'models/churn_model.pkl')
   
   # Na API:
   model = joblib.load('models/churn_model.pkl')
   prediction = model.predict(features)
"""

# ============================================================
# PLACEHOLDER - Modelo mockado para desenvolvimento
# ============================================================

class MockModel:
    """Modelo placeholder para desenvolvimento sem arquivo real."""
    
    def __init__(self):
        self.is_loaded = True
    
    def predict(self, data):
        """Retorna predição mockada."""
        # Lógica simples para desenvolvimento
        return [0]
    
    def predict_proba(self, data):
        """Retorna probabilidades mockadas."""
        return [[0.3, 0.7]]


def get_mock_model() -> MockModel:
    """Retorna instância do modelo mockado."""
    return MockModel()