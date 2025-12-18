import pickle
import numpy as np
import pytest
from pathlib import Path
from fivedreg.data import load_dataset, split_data, standardize_features

# --- Tests for load_dataset ---
def test_load_dataset_no_y(tmp_path):
    # Create dataset without 'y'
    data = {"X": np.zeros((10, 5))}
    p = tmp_path / "noy.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)
    
    X, y = load_dataset(p)
    assert X.shape == (10, 5)
    assert y is None

def test_load_dataset_invalid_extension(tmp_path):
    p = tmp_path / "data.txt"
    p.touch()
    with pytest.raises(ValueError, match="Expected .pkl file"):
        load_dataset(p)

def test_load_dataset_bad_structure(tmp_path):
    p = tmp_path / "bad.pkl"
    with open(p, "wb") as f:
        pickle.dump(["not", "a", "dict"], f)
    with pytest.raises(ValueError, match="Expected a dict"):
        load_dataset(p)

def test_load_dataset_bad_shapes(tmp_path):
    # Wrong X dim
    p = tmp_path / "bad_X.pkl"
    with open(p, "wb") as f:
        pickle.dump({"X": np.zeros((10, 3))}, f) # Only 3 features
    with pytest.raises(ValueError, match="Expected 5 features"):
        load_dataset(p)

# --- Tests for split_data ---

def test_split_data_shapes():
    X = np.zeros((100, 5))
    y = np.zeros(100)
    
    # 20% test -> 20 items. 80 remaining. 
    # 20% val of remaining 80 -> 16 items.
    # Train = 100 - 20 - 16 = 64.
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.2, random_state=42
    )
    
    assert len(X_test) == 20
    assert len(X_val) == 16
    assert len(X_train) == 64
    assert len(y_train) == 64

def test_split_data_no_y():
    X = np.zeros((100, 5))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, None, test_size=0.2, val_size=0.2
    )
    assert len(X_train) == 64
    assert y_train is None
    assert y_val is None
    assert y_test is None

# --- Tests for standardize_features ---

def test_standardize_features_statistics():
    # Construct data with known stats
    # Train: mean=10, std=2
    # Val/Test: should be transformed by Train's stats
    X_train = np.random.normal(loc=10.0, scale=2.0, size=(1000, 5))
    X_val = np.array([[10.0]*5, [12.0]*5]) # Should map to ~0 and ~1
    X_test = np.copy(X_val)
    
    X_train_s, X_val_s, X_test_s, _, scaler = standardize_features(X_train, X_val, X_test)
    
    # Check train standardization
    assert np.allclose(X_train_s.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(X_train_s.std(axis=0), 1.0, atol=0.1)
    
    # Check val transformation using fitted scaler
    # 10.0 -> (10-10)/2 = 0
    # 12.0 -> (12-10)/2 = 1
    assert np.allclose(X_val_s[0], 0.0, atol=0.1)
    assert np.allclose(X_val_s[1], 1.0, atol=0.1)


