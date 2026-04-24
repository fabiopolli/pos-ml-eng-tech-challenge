import pytest

from src.models.config import MLFlowConfig, MLPConfig, PipelineConfig
from src.models.train_baselines import main as train_baselines
from src.models.train_mlp import main as train_mlp


# Configuração compartilhada que desabilita o Model Registry nos testes.
# O tracking ainda acontece (no diretório temporário do conftest.py),
# mas não tentamos registrar modelos em um Registry que não existe nos testes.
# epochs=2 torna os smoke tests muito mais rápidos (ao invés de 50 épocas).
_TEST_CONFIG = PipelineConfig(
    mlp=MLPConfig(epochs=2, batch_size=32),
    mlflow=MLFlowConfig(register_model=False),
)



@pytest.mark.slow
def test_train_baselines_smoke(sample_csv_path, tmp_path):
    """train_baselines deve rodar sem erros e salvar os 3 artefatos esperados."""
    train_baselines(data_path=sample_csv_path, models_dir=tmp_path, config=_TEST_CONFIG)

    assert (tmp_path / "dummy_model.pkl").exists(), "dummy_model.pkl não gerado"
    assert (tmp_path / "logistic_model.pkl").exists(), "logistic_model.pkl não gerado"
    assert (tmp_path / "scaler.pkl").exists(), "scaler.pkl não gerado"


@pytest.mark.slow
def test_train_mlp_smoke(sample_csv_path, tmp_path):
    """train_mlp deve rodar sem erros e salvar o modelo."""
    train_mlp(data_path=sample_csv_path, models_dir=tmp_path, config=_TEST_CONFIG)

    assert (tmp_path / "mlp_model.pth").exists(), "mlp_model.pth não gerado"


@pytest.mark.slow
def test_pipeline_completo_smoke(sample_csv_path, tmp_path):
    """Pipeline completo (baselines + mlp) deve produzir todos os 4 artefatos."""
    train_baselines(data_path=sample_csv_path, models_dir=tmp_path, config=_TEST_CONFIG)
    train_mlp(data_path=sample_csv_path, models_dir=tmp_path, config=_TEST_CONFIG)

    artefatos = ["dummy_model.pkl", "logistic_model.pkl", "scaler.pkl", "mlp_model.pth"]
    for artefato in artefatos:
        assert (tmp_path / artefato).exists(), f"Artefato '{artefato}' não encontrado"


@pytest.mark.slow
def test_train_baselines_arquivo_inexistente(tmp_path):
    """train_baselines deve retornar sem levantar exceção quando o dataset não existe."""
    caminho_invalido = tmp_path / "nao_existe.csv"
    # Não deve levantar exceção — deve logar o erro e retornar
    train_baselines(data_path=caminho_invalido, models_dir=tmp_path)
    # Nenhum artefato deve ser gerado
    assert not (tmp_path / "dummy_model.pkl").exists()
