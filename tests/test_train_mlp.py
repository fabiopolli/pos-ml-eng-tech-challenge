import torch
import torch.nn as nn

from src.models.train_mlp import ChurnMLP


def test_output_shape_padrao():
    model = ChurnMLP(input_dim=10)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 1), f"Shape esperado (8, 1), obtido {out.shape}"


def test_output_shape_batch_unitario():
    model = ChurnMLP(input_dim=10)
    x = torch.randn(1, 10)
    assert model(x).shape == (1, 1)


def test_hidden_dims_customizados():
    model = ChurnMLP(input_dim=10, hidden_dims=[64, 32, 16])
    x = torch.randn(4, 10)
    assert model(x).shape == (4, 1)


def test_hidden_dims_none_usa_padrao():
    """hidden_dims=None deve resultar em arquitetura [32, 16]."""
    model = ChurnMLP(input_dim=5, hidden_dims=None)
    linear_layers = [m for m in model.network if isinstance(m, nn.Linear)]
    # 3 camadas lineares: 5→32, 32→16, 16→1
    assert len(linear_layers) == 3
    assert linear_layers[0].out_features == 32
    assert linear_layers[1].out_features == 16
    assert linear_layers[2].out_features == 1


def test_sem_dropout_por_padrao():
    model = ChurnMLP(input_dim=10)
    assert not any(isinstance(m, nn.Dropout) for m in model.network)


def test_com_dropout_inserido():
    model = ChurnMLP(input_dim=10, dropout_rate=0.3)
    dropout_layers = [m for m in model.network if isinstance(m, nn.Dropout)]
    assert len(dropout_layers) > 0
    assert dropout_layers[0].p == 0.3


def test_forward_gradiente_nao_explode():
    """Verifica estabilidade numérica: saída deve ser finita."""
    model = ChurnMLP(input_dim=20)
    x = torch.randn(16, 20)
    out = model(x)
    assert torch.isfinite(out).all(), "Saída contém NaN ou Inf"
