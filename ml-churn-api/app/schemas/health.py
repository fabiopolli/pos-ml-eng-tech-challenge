from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    status: str
    version: str
    database: Optional[str] = None


class HealthDetail(BaseModel):
    """Schema para detalhes do health check."""
    api: str
    model: Optional[str] = None
    database: Optional[str] = None