import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configurações da aplicação."""
    
    # App
    app_name: str = "ML Churn API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API
    api_prefix: str = "/api/v1"
    
    # Model
    model_path: str = "models/logistic_model.pkl"  # Arquivo físico de fallback
    model_version: str = "1.0.0"
    
    # MLflow
    mlflow_tracking_uri: str = "mlruns"
    mlflow_model_name: str = "ChurnLogisticRegression"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Retorna as configurações cacheadas."""
    return Settings()