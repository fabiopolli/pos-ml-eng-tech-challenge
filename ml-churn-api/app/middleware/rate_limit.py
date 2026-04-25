"""
Middleware para tracking de rate limit (placeholder).

Este middleware pode ser expandido para implementar
limitação de requisições por IP ou usuário.
"""

import time
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware para limitação de requisições.
    
    Implementação básica:
    - Limita por IP
    - 100 requisições por minuto (configurável)
    
    Para produção, considere usar:
    - slowapi (https://github.com/tiangolo/fastapi-slowapi)
    - redis para storage distribuído
    """
    
    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        # Obter IP do cliente
        client_ip = request.client.host if request.client else "unknown"
        
        # Verificar rate limit
        current_time = time.time()
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if current_time - req_time < 60  # Limpar requisições mais antigas
        ]
        
        # Verificar se excedeu o limite
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit excedido para IP: {client_ip}")
            return Response(
                content='{"error": "Rate limit excedido"}',
                status_code=429,
                media_type="application/json",
            )
        
        # Adicionar requisição atual
        self.requests[client_ip].append(current_time)
        
        # Processar requisição
        response = await call_next(request)
        
        # Adicionar headers de rate limit
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.requests[client_ip])
        )
        
        return response