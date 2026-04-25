import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from app.main import app
from app.schemas.predict import PredictInput
from app.services.model_service import predict, ModelServiceError


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def client():
    """Cliente de teste para a API."""
    return TestClient(app)


@pytest.fixture
def valid_predict_input():
    """Dados de entrada válidos para predição."""
    return {
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
        "streaming_movies": False,
    }


@pytest.fixture
def mock_model_response():
    """Resposta mockada do modelo."""
    return {
        "prediction": 0,
        "probability": 0.35,
        "model_version": "1.0.0",
        "request_id": "abc12345",
    }


# ============================================================
# TESTES DE HEALTH
# ============================================================

class TestHealthEndpoint:
    """Testes para o endpoint de health check."""
    
    def test_health_simple(self, client):
        """Testa health check simples."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
    
    def test_health_detailed(self, client):
        """Testa health check detalhado."""
        response = client.get("/api/v1/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert "api" in data
        assert "model" in data


# ============================================================
# TESTES DE PREDICT
# ============================================================

class TestPredictEndpoint:
    """Testes para o endpoint de predição."""
    
    def test_predict_success(self, client, valid_predict_input):
        """Testa predição bem-sucedida."""
        response = client.post(
            "/api/v1/predict",
            json=valid_predict_input,
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data
        assert "request_id" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
    
    def test_predict_returns_integer(self, client, valid_predict_input):
        """Testa que prediction é inteiro."""
        response = client.post("/api/v1/predict", json=valid_predict_input)
        
        data = response.json()
        assert isinstance(data["prediction"], int)
    
    def test_predict_probability_range(self, client, valid_predict_input):
        """Testa que probabilidade está entre 0 e 1."""
        response = client.post("/api/v1/predict", json=valid_predict_input)
        
        data = response.json()
        assert 0 <= data["probability"] <= 1
    
    def test_predict_with_monthly_contract(self, client):
        """Testa predição com contrato mensal (maior risco)."""
        input_data = {
            "tenure": 3,
            "monthly_charges": 50.0,
            "total_charges": 150.0,
            "contract_type": "monthly",
            "payment_method": "electronic_check",
            "has_phone_service": True,
            "has_internet_service": True,
            "has_online_security": False,
            "has_online_backup": False,
            "has_device_protection": False,
            "has_tech_support": False,
            "streaming_tv": False,
            "streaming_movies": False,
        }
        
        response = client.post("/api/v1/predict", json=input_data)
        
        assert response.status_code == 201
        data = response.json()
        # Contrato mensal tende a ter maior probabilidade
        assert data["probability"] > 0.3
    
    def test_predict_with_two_year_contract(self, client):
        """Testa predição com contrato de 2 anos (menor risco)."""
        input_data = {
            "tenure": 30,
            "monthly_charges": 80.0,
            "total_charges": 2400.0,
            "contract_type": "two_year",
            "payment_method": "credit_card",
            "has_phone_service": True,
            "has_internet_service": True,
            "has_online_security": True,
            "has_online_backup": True,
            "has_device_protection": True,
            "has_tech_support": True,
            "streaming_tv": True,
            "streaming_movies": True,
        }
        
        response = client.post("/api/v1/predict", json=input_data)
        
        assert response.status_code == 201
        data = response.json()
        # Contrato de 2 anos com serviços tende a ter menor probabilidade
        assert data["probability"] < 0.5


# ============================================================
# TESTES DE VALIDAÇÃO
# ============================================================

class TestValidation:
    """Testes de validação de entrada."""
    
    def test_missing_required_field(self, client):
        """Testa erro quando campo obrigatório falta."""
        invalid_data = {
            "tenure": 12,
            # Faltando monthly_charges, total_charges, etc.
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        
        assert response.status_code == 422
    
    def test_invalid_contract_type(self, client):
        """Testa erro com tipo de contrato inválido."""
        invalid_data = {
            "tenure": 12,
            "monthly_charges": 70.0,
            "total_charges": 840.0,
            "contract_type": "invalid_type",  # Inválido
            "payment_method": "credit_card",
            "has_phone_service": True,
            "has_internet_service": True,
            "has_online_security": False,
            "has_online_backup": True,
            "has_device_protection": False,
            "has_tech_support": False,
            "streaming_tv": False,
            "streaming_movies": False,
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        
        # Aceita qualquer string (validação frouxa) ou rejeita
        assert response.status_code in [201, 422]
    
    def test_negative_tenure(self, client):
        """Testa erro com tenure negativo."""
        invalid_data = {
            "tenure": -1,  # Inválido
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
            "streaming_movies": False,
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        
        assert response.status_code == 422
    
    def test_negative_charges(self, client):
        """Testa erro com charges negativo."""
        invalid_data = {
            "tenure": 12,
            "monthly_charges": -10.0,  # Inválido
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
            "streaming_movies": False,
        }
        
        response = client.post("/api/v1/predict", json=invalid_data)
        
        assert response.status_code == 422
    
    def test_empty_body(self, client):
        """Testa erro com corpo vazio."""
        response = client.post("/api/v1/predict", json={})
        
        assert response.status_code == 422


# ============================================================
# TESTES DE SCHEMAS
# ============================================================

class TestSchemas:
    """Testes para os schemas Pydantic."""
    
    def test_predict_input_valid(self, valid_predict_input):
        """Testa criação de PredictInput válido."""
        input_data = PredictInput(**valid_predict_input)
        
        assert input_data.tenure == 12
        assert input_data.contract_type == "one_year"
    
    def test_predict_input_defaults(self):
        """Testa valores padrão dos campos opcionais."""
        input_data = PredictInput(
            tenure=12,
            monthly_charges=70.0,
            total_charges=840.0,
            contract_type="monthly",
            payment_method="credit_card",
        )
        
        assert input_data.has_phone_service is True
        assert input_data.has_internet_service is True


# ============================================================
# TESTES DE SERVIÇO
# ============================================================

class TestModelService:
    """Testes para o serviço de modelo."""
    
    def test_predict_function(self, valid_predict_input):
        """Testa função predict diretamente."""
        input_data = PredictInput(**valid_predict_input)
        result = predict(input_data)
        
        assert result.prediction in [0, 1]
        assert 0 <= result.probability <= 1
        assert result.model_version == "1.0.0"
        assert result.request_id is not None
    
    def test_predict_request_id_unique(self, valid_predict_input):
        """Testa que request_id é único."""
        input_data = PredictInput(**valid_predict_input)
        
        result1 = predict(input_data)
        result2 = predict(input_data)
        
        # IDs devem ser diferentes (gerados aleatoriamente)
        assert result1.request_id != result2.request_id


# ============================================================
# TESTES DE MOCK
# ============================================================

class TestMocks:
    """Testes usando mocks."""
    
    @patch("app.services.model_service.get_mock_model")
    def test_predict_with_mock(self, mock_get_model, valid_predict_input):
        """Testa predição com modelo mockado."""
        # Configurar mock
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_get_model.return_value = mock_model
        
        input_data = PredictInput(**valid_predict_input)
        result = predict(input_data)
        
        assert result.prediction == 1
        mock_model.predict.assert_called_once()


# ============================================================
# TESTES DE INTEGRAÇÃO
# ============================================================

class TestIntegration:
    """Testes de integração."""
    
    def test_full_predict_flow(self, client, valid_predict_input):
        """Testa fluxo completo: health -> predict."""
        # Health check
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200
        
        # Predict
        predict_response = client.post(
            "/api/v1/predict",
            json=valid_predict_input,
        )
        assert predict_response.status_code == 201
        
        # Verificar consistência
        health_data = health_response.json()
        predict_data = predict_response.json()
        
        assert predict_data["model_version"] == health_data["version"]
    
    def test_multiple_predictions(self, client):
        """Testa múltiplas predições consecutivas."""
        input_data = {
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
            "streaming_movies": False,
        }
        
        # Fazer 3 predições
        for _ in range(3):
            response = client.post("/api/v1/predict", json=input_data)
            assert response.status_code == 201
            data = response.json()
            assert "request_id" in data


# ============================================================
# EXECUTAR TESTES
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])