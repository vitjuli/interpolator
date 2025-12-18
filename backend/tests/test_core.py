import numpy as np
import pytest
import torch
from fivedreg.model import FiveDRegressor

# --- Fixtures ---

@pytest.fixture
def clean_model():
    """Returns a fresh model instance with fixed seed."""
    return FiveDRegressor(random_state=42, hidden_layers=(16, 8))

@pytest.fixture
def dummy_data():
    """Returns synthetic (X, y) for 5D regression."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 5)).astype(np.float32)
    # y = x0 + x1
    y = (X[:, 0] + X[:, 1]).astype(np.float32)
    return X, y

# --- Initialization Tests ---

def test_model_init_defaults():
    model = FiveDRegressor()
    assert model.learning_rate == 1e-3
    assert not model._is_fitted
    assert len(model.network) > 0

def test_model_structure():
    model = FiveDRegressor(hidden_layers=(10, 5))
    # Input(5)->10 (Linear+ReLU) -> 5 (Linear+ReLU) -> 1 (Linear)
    # Layers: Lin(5,10), ReLU, Lin(10,5), ReLU, Lin(5,1)
    # Total modules in sequential: 5
    assert len(model.network) == 5
    assert isinstance(model.network[0], torch.nn.Linear)
    assert model.network[0].in_features == 5
    assert model.network[0].out_features == 10

# --- Validation Tests ---

def test_check_xy_shapes(clean_model):
    # Correct consistency
    X = np.zeros((10, 5))
    y = np.zeros(10)
    clean_model._check_Xy(X, y)

    # Wrong X dim
    with pytest.raises(ValueError, match="Expected X shape"):
        clean_model._check_Xy(np.zeros((10, 4)), y)
    
    # Wrong y length
    with pytest.raises(ValueError, match="X and y must have same length"):
        clean_model._check_Xy(X, np.zeros(11))

def test_check_xy_nans(clean_model):
    X = np.zeros((10, 5))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="X contains NaN"):
        clean_model._check_Xy(X)

# --- Training Logic Tests ---



def test_predict_unfitted_raises(clean_model, dummy_data):
    X, _ = dummy_data
    with pytest.raises(RuntimeError, match="not fitted"):
        clean_model.predict(X)

def test_reproducibility(dummy_data):
    X, y = dummy_data
    
    # Train two models with same seed
    m1 = FiveDRegressor(random_state=123, max_epochs=5)
    m1.fit(X, y)
    p1 = m1.predict(X)
    
    m2 = FiveDRegressor(random_state=123, max_epochs=5)
    m2.fit(X, y)
    p2 = m2.predict(X)
    
    assert np.allclose(p1, p2)

def test_early_stopping_logic(dummy_data):
    # This is a bit tricky to test deterministically without mocking,
    # but we can check if it runs without error when validation data is provided.
    X, y = dummy_data
    X_val, y_val = X[:10], y[:10]
    X_train, y_train = X[10:], y[10:]
    
    model = FiveDRegressor(patience=2, min_delta=1000.0, max_epochs=10, verbose=True) 
    # High min_delta impossible to satisfy -> should stop early
    
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    # We can't easily assert the exact epoch it stopped without capturing stdout/logging
    # or inspecting internal state, but we ensure it finishes and is fitted.
    assert model._is_fitted
