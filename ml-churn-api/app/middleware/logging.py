"""
Middleware para logging de requisições HTTP.

Registra informações sobre cada requisição recebida.
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para log de requisições e respostas.
    
    Registra:
    - Método HTTP
    - Path solicitado
    - Status code da resposta
    - Tempo de processamento
    """
    
    async def dispatch(self, request: Request, call_next):
        # Iniciar timer
        start_time = time.time()
        
        # Log da requisição recebida
        logger.info(
            f"Requisição recebida: {request.method} {request.url.path}"
        )
        
        # Processar requisição
        response = await call_next(request)
        
        # Calcular tempo de processamento
        process_time = time.time() - start_time
        
        # Log da resposta
        logger.info(
            f"Resposta: {request.method} {request.url.path} "
            f"status={response.status_code} time={process_time:.3f}s"
        )
        
        # Adicionar header com tempo de processamento
        response.headers["X-Process-Time"] = str(process_time)
        
        return response