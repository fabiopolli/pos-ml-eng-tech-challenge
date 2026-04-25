"""
Middleware para adicionar headers de segurança e cache.

Headers recomendados para APIs REST.
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware para adicionar headers de segurança.
    
    Adiciona:
    - X-Request-ID: ID único para cada requisição
    - X-Content-Type-Options: Previne MIME type sniffing
    - X-Frame-Options: Proteção contra clickjacking
    """
    
    async def dispatch(self, request: Request, call_next):
        # Gerar ID único para a requisição
        request_id = f"{int(time.time() * 1000)}"
        
        # Processar requisição
        response = await call_next(request)
        
        # Adicionar headers de segurança
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Headers de cache (pode ser ajustado por endpoint)
        if request.method == "GET":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        else:
            response.headers["Cache-Control"] = "no-store, max-age=0"
        
        return response