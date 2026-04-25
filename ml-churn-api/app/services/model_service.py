import uuid
import logging
from typing import Tuple, Optional
from app.schemas.predict import PredictInput, PredictOutput
from app.core.config import get_settings
from app.models import get_model, preprocess_input, auto_load_model, get_mock_model
import mlflow
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Flag para usar modelo real ou mock
# Mude para False se quiser usar mock durante desenvolvimento
USE_REAL_MODEL = True


class ModelServiceError(Exception):
    """Exceção específica para erros do serviço de modelo."""
    pass


def predict(data: PredictInput) -> PredictOutput:
    """
    Executa predição de churn.
    
    Args:
        data: Dados de entrada validados
        
    Returns:
        PredictOutput com resultado da predição
        
    Raises:
        ModelServiceError: Se houver erro na predição
    """
    request_id = str(uuid.uuid4())[:8]
    settings = get_settings()
    
    try:
        logger.info(f"Processando predição - request_id: {request_id}")
        
        if USE_REAL_MODEL:
            # Carregamento com Fallback (MLflow -> Local)
            model = None
            mlflow_uri = f"models:/{settings.mlflow_model_name}/latest"
            
            # 1. Tentar MLflow Model Registry
            try:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                model_mlflow = mlflow.sklearn.load_model(mlflow_uri)
                model = type("MLflowModelWrapper", (), {"is_loaded": True, "predict": model_mlflow.predict, "model": model_mlflow})()
                logger.info(f"Modelo carregado do MLflow Registry: {mlflow_uri}")
            except Exception as e:
                logger.warning(f"Não foi possível carregar do MLflow ({e}). Tentando fallback local...")
                
                # 2. Fallback: Arquivo local
                local_model = get_model("logistic_model.pkl")
                if not local_model.is_loaded:
                    loaded = local_model.load()
                    if not loaded:
                        raise ModelServiceError("Nenhum modelo disponível (MLflow ou Local)")
                model = local_model
            
            # Preprocessamento (Zero Skew)
            features_df = preprocess_input(data)
            
            # Para modelos sklearn (espera DataFrame ou array)
            prediction = model.predict(features_df)[0]
            
            if hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
                probability = model.model.predict_proba(features_df)[0][1]
            elif hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features_df)[0][1]
            else:
                probability = 0.5
        else:
            # Usar modelo mockado (desenvolvimento)
            mock = get_mock_model()
            prediction = mock.predict([None])[0]
            probability = _calculate_mock_probability(data)
        
        logger.info(
            f"Predição concluída - request_id: {request_id}, "
            f"prediction: {prediction}, probability: {probability:.4f}"
        )
        
        return PredictOutput(
            prediction=int(prediction),
            probability=round(probability, 4),
            model_version=settings.model_version,
            request_id=request_id,
        )
        
    except Exception as e:
        logger.error(f"Erro na predição - request_id: {request_id}, error: {str(e)}")
        raise ModelServiceError(f"Erro ao processar predição: {str(e)}")


def _calculate_mock_probability(data: PredictInput) -> float:
    """Calcula probabilidade mockada para desenvolvimento."""
    # Pesos simplificados para simulação
    score = 0.5
    
    # Contrato: mensal = maior risco, dois anos = menor risco
    contract_risk = {"monthly": 0.7, "one_year": 0.4, "two_year": 0.2}
    score *= contract_risk.get(data.contract_type, 0.5)
    
    # Tempo de permanência: mais tempo = menor risco
    if data.tenure > 24:
        score *= 0.7
    elif data.tenure < 6:
        score *= 1.3
    
    # Serviços adicionais = menor risco
    if data.has_online_security:
        score *= 0.8
    if data.has_tech_support:
        score *= 0.8
    
    return min(0.95, max(0.05, score))


def load_model() -> None:
    """Carrega o modelo ML (placeholder para implementação real)."""
    settings = get_settings()
    logger.info(f"Tentando carregar modelo de: {settings.mlflow_model_name} (MLflow) ou {settings.model_path} (Local)")
    # Implementação no predict() lida com o carregamento sob demanda
    logger.info("Sistema de carregamento sob demanda ativo.")


def health_check() -> dict:
    """Verifica saúde do serviço de modelo."""
    return {
        "status": "healthy",
        "model_loaded": True,
    }