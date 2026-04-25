import uuid
import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class APIException(Exception):
    """Exceção base para erros da API."""
    
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ModelServiceException(APIException):
    """Exceção para erros do serviço de modelo."""
    
    def __init__(self, message: str = "Erro no serviço de modelo"):
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)


class ValidationException(APIException):
    """Exceção para erros de validação."""
    
    def __init__(self, message: str = "Erro de validação"):
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY)


async def api_exception_handler(request: Request, exc: APIException):
    """Handler para exceções customizadas da API."""
    request_id = str(uuid.uuid4())[:8]
    
    logger.error(
        f"API Exception - request_id: {request_id}, "
        f"path: {request.url.path}, error: {exc.message}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "request_id": request_id,
            "path": str(request.url),
        },
    )


async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handler para erros de validação do Pydantic."""
    request_id = str(uuid.uuid4())[:8]
    
    logger.warning(
        f"Validation Error - request_id: {request_id}, "
        f"path: {request.url.path}, errors: {exc.errors()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Erro de validação nos dados enviados",
            "details": exc.errors(),
            "request_id": request_id,
        },
    )


async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler para erros de validação do FastAPI."""
    request_id = str(uuid.uuid4())[:8]
    
    logger.warning(
        f"Request Validation Error - request_id: {request_id}, "
        f"path: {request.url.path}, errors: {exc.errors()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "RequestValidationError",
            "message": "Erro de validação na requisição",
            "details": exc.errors(),
            "request_id": request_id,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handler genérico para exceções não tratadas."""
    request_id = str(uuid.uuid4())[:8]
    
    logger.error(
        f"Unhandled Exception - request_id: {request_id}, "
        f"path: {request.url.path}, error: {type(exc).__name__}: {str(exc)}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "Erro interno no servidor",
            "request_id": request_id,
        },
    )