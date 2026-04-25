import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routes import health, predict
from app.exceptions.handlers import (
    api_exception_handler,
    validation_exception_handler,
    request_validation_exception_handler,
    generic_exception_handler,
)
from app.exceptions import APIException, ValidationException
from app.middleware import (
    LoggingMiddleware,
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
)
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    # Startup
    logger.info("Iniciando ML Churn API...")
    settings = get_settings()
    logger.info(f"Versão: {settings.app_version}")
    
    # Carregar modelo ML
    from app.services import load_model
    load_model()
    
    yield
    
    # Shutdown
    logger.info("Encerrando ML Churn API...")


# Criar aplicação FastAPI
settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API para predição de churn de clientes",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom Middlewares (ordem: executados de baixo para cima)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(LoggingMiddleware)
# Descomente para ativar rate limit:
# app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

# Exception Handlers
app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(ValidationException, validation_exception_handler)
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Rotas
app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(predict.router, prefix=settings.api_prefix)