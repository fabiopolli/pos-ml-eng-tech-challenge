import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.schemas.predict import PredictInput, PredictOutput
from app.services import predict as predict_service
from app.services.model_service import ModelServiceError
from app.exceptions import ModelServiceException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"], prefix="/predict")


@router.post(
    "",
    response_model=PredictOutput,
    status_code=status.HTTP_201_CREATED,
    summary="Executar predição de churn",
    description="Recebe dados do cliente e retorna a predição de churn",
    responses={
        201: {"description": "Predição executada com sucesso"},
        422: {"description": "Erro de validação nos dados"},
        503: {"description": "Erro no serviço de modelo"},
    },
)
def predict_route(data: PredictInput):
    """
    Endpoint para predição de churn.
    
    Args:
        data: Dados de entrada do cliente (validados pelo Pydantic)
        
    Returns:
        PredictOutput com prediction, probability, model_version e request_id
    """
    try:
        logger.info(f"Recebida requisição de predição para tenure={data.tenure}")
        
        result = predict_service(data)
        
        logger.info(f"Predição concluída: {result.dict()}")
        return result
        
    except ModelServiceError as e:
        logger.error(f"Erro no serviço de modelo: {str(e)}")
        raise ModelServiceException(str(e))