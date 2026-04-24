import numpy as np

from src.models.data_utils import get_data_splits, set_seed


def test_get_data_splits_retorna_7_elementos(sample_csv_path):
    result = get_data_splits(sample_csv_path)
    assert len(result) == 7


def test_splits_sem_overlap_de_indices(sample_csv_path):
    X_train, _, X_val, _, X_test, _, _ = get_data_splits(sample_csv_path)
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(val_idx), "Treino e Validação têm índices em comum"
    assert train_idx.isdisjoint(test_idx), "Treino e Teste têm índices em comum"
    assert val_idx.isdisjoint(test_idx), "Validação e Teste têm índices em comum"


def test_sem_data_leakage_no_scaler(sample_csv_path):
    """Média do treino deve ser ~0; val e test não podem ser exatamente 0."""
    X_train, _, X_val, _, X_test, _, _ = get_data_splits(sample_csv_path)
    assert abs(X_train["tenure"].mean()) < 0.5
    # Val e Test são escalados com as stats do treino → média ≠ 0
    assert abs(X_val["tenure"].mean()) != 0.0 or abs(X_test["tenure"].mean()) != 0.0


def test_set_seed_reprodutibilidade():
    set_seed(123)
    a = np.random.rand()
    set_seed(123)
    b = np.random.rand()
    assert a == b, "set_seed não garante reprodutibilidade"


def test_set_seed_valores_diferentes_para_seeds_distintas():
    set_seed(1)
    a = np.random.rand()
    set_seed(2)
    b = np.random.rand()
    assert a != b
