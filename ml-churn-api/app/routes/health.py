import logging
from fastapi import APIRouter, status
from app.schemas.health import HealthResponse, HealthDetail
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"], prefix="/health")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health check simples",
    description="Retorna o status da API",
)
def health():
    """Endpoint simples de health check."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
    )


@router.get(
    "/detailed",
    response_model=HealthDetail,
    summary="Health check detalhado",
    description="Retorna o status detalhado de todos os componentes",
)
def health_detailed():
    """Endpoint de health check com detalhes dos componentes."""
    settings = get_settings()
    
    # Verificar componentes
    from app.services import health_check as model_health
    
    model_status = model_health()
    
    return HealthDetail(
        api=f"{settings.app_name} v{settings.app_version}",
        model=model_status.get("status"),
        database=None,
    )