from pydantic import BaseModel, Field
from typing import Optional


class PredictInput(BaseModel):
    """Schema para dados de entrada da predição."""
    tenure: int = Field(..., ge=0, description="Tempo de permanência do cliente em meses")
    monthly_charges: float = Field(..., ge=0, description="Valor da mensalidade")
    total_charges: float = Field(..., ge=0, description="Valor total gasto")
    contract_type: str = Field(..., description="Tipo de contrato: monthly, one_year, two_year")
    payment_method: str = Field(..., description="Método de pagamento")
    has_phone_service: bool = Field(default=True, description="Possui serviço de telefone")
    has_internet_service: bool = Field(default=True, description="Possui serviço de internet")
    has_online_security: bool = Field(default=False, description="Possui segurança online")
    has_online_backup: bool = Field(default=False, description="Possui backup online")
    has_device_protection: bool = Field(default=False, description="Possui proteção de dispositivo")
    has_tech_support: bool = Field(default=False, description="Possui suporte técnico")
    streaming_tv: bool = Field(default=False, description="Assiste TV via streaming")
    streaming_movies: bool = Field(default=False, description="Assiste filmes via streaming")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "monthly_charges": 70.0,
                "total_charges": 840.0,
                "contract_type": "one_year",
                "payment_method": "credit_card",
                "has_phone_service": True,
                "has_internet_service": True,
                "has_online_security": False,
                "has_online_backup": True,
                "has_device_protection": False,
                "has_tech_support": False,
                "streaming_tv": False,
                "streaming_movies": False
            }
        }


class PredictOutput(BaseModel):
    """Schema para resposta da predição."""
    prediction: int = Field(..., description="0: Não vai cancelar, 1: Vai cancelar")
    probability: Optional[float] = Field(None, description="Probabilidade de churn (0-1)")
    model_version: str = Field(default="1.0.0", description="Versão do modelo")
    request_id: str = Field(..., description="ID único da requisição")


class PredictError(BaseModel):
    """Schema para erros."""
    error: str
    detail: str
    request_id: Optional[str] = None