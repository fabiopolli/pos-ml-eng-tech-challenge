"""
Wrapper para expor o modelo PyTorch como um modelo compatível com a API.

Este módulo carrega o modelo PyTorch (.pth) e o expõe com uma interface
compatível com o padrão sklearn (predict/predict_proba) que a API espera.
"""

import os
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Caminhos
BASE_DIR = Path(__file__).resolve().parents[3]  # raiz do projeto
MODELS_DIR = BASE_DIR / "models"
PTH_MODEL_PATH = MODELS_DIR / "mlp_model.pth"
PKL_MODEL_PATH = MODELS_DIR / "churn_model.pkl"


class PyTorchModelWrapper:
    """
    Wrapper que carrega um modelo PyTorch e o expõe como sklearn-compatible.
    
    Permite que a API FastAPI (que espera um modelo .pkl) use o modelo .pth.
    """
    
    def __init__(self, pth_path: str = None):
        self.pth_path = pth_path or str(PTH_MODEL_PATH)
        self.model = None
        self.input_dim = None
        self._metadata = {}
    
    def load(self) -> bool:
        """Carrega o modelo PyTorch e seus metadados."""
        try:
            if not os.path.exists(self.pth_path):
                logger.warning(f"Modelo PyTorch não encontrado: {self.pth_path}")
                return False
            
            # Carrega metadados se existirem
            metadata_path = self.pth_path.replace(".pth", "_metadata.pkl")
            if os.path.exists(metadata_path):
                self._metadata = joblib.load(metadata_path)
                self.input_dim = self._metadata.get("input_dim", 18)
            else:
                # Default input_dim para este dataset
                self.input_dim = 18
            
            # Reconstrói a arquitetura do modelo
            self.model = ChurnMLPWrapper(input_dim=self.input_dim)
            self.model.load_state_dict(torch.load(self.pth_path, map_location="cpu"))
            self.model.eval()
            
            logger.info(f"Modelo PyTorch carregado: {self.pth_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo PyTorch: {e}")
            return False
    
    def predict(self, X: Any) -> np.ndarray:
        """
        Predição binária (sklearn-compatible).
        
        Args:
            X: Array de features
            
        Returns:
            Array de predições (0 ou 1)
        """
        if self.model is None:
            raise RuntimeError("Modelo não carregado")
        
        # Converte para tensor se necessário
        if isinstance(X, list):
            X = np.array(X)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = (torch.sigmoid(logits) > 0.5).int().numpy()
        
        return predictions.flatten()
    
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predição com probabilidades (sklearn-compatible).
        
        Args:
            X: Array de features
            
        Returns:
            Array de probabilidades [P(0), P(1)]
        """
        if self.model is None:
            raise RuntimeError("Modelo não carregado")
        
        if isinstance(X, list):
            X = np.array(X)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).numpy()
        
        # Formato sklearn: [prob_class_0, prob_class_1]
        return np.column_stack([1 - probs, probs])


class ChurnMLPWrapper(nn.Module):
    """Arquitetura idêntica ao modelo treinado."""
    
    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 16]
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def convert_pth_to_pkl() -> bool:
    """
    Converte o modelo .pth para .pkl (wrapper sklearn-compatible).
    
    Returns:
        bool: True se conversão bem-sucedida
    """
    try:
        wrapper = PyTorchModelWrapper()
        
        if not wrapper.load():
            logger.error("Falha ao carregar modelo PyTorch")
            return False
        
        # Salva o wrapper como .pkl
        joblib.dump(wrapper, str(PKL_MODEL_PATH))
        logger.info(f"Modelo convertido: {PKL_MODEL_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"Erro na conversão: {e}")
        return False


if __name__ == "__main__":
    # Executar conversão
    logging.basicConfig(level=logging.INFO)
    success = convert_pth_to_pkl()
    print(f"Conversão {'sucesso' if success else 'falhou'}")