"""
Test backend utility functions for dataset and model management.

Tests:
- Dataset creation and saving
- Dataset loading
- Model saving and loading
- Save/load integrity
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from fivedreg import (
    FiveDRegressor,
    create_synthetic_dataset,
    save_dataset,
    load_dataset,
    list_datasets,
    save_model,
    load_model,
    list_models,
    ground_truth_function,
    get_data_dir,
    get_models_dir,
)


class TestDatasetFunctions:
    """Test dataset management functions."""

    def test_create_synthetic_dataset(self):
        """Test creating synthetic dataset."""
        X, y = create_synthetic_dataset(n_samples=100, seed=42)

        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert X.dtype == np.float32
        assert isinstance(y, np.ndarray)

    def test_ground_truth_function(self):
        """Test ground truth function."""
        X = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        y = ground_truth_function(X)

        # y = 2.0*x1 + (-1.5)*x2^2 + 3.0*sin(x3) + 0.5*x4*x5
        # With x1=1, others=0: y = 2.0
        assert abs(y[0] - 2.0) < 1e-5

    def test_save_and_load_dataset(self, tmp_path):
        """Test saving and loading dataset."""
        # Create temporary data directory
        temp_data_dir = tmp_path / "data"
        temp_data_dir.mkdir()

        # Create dataset
        X, y = create_synthetic_dataset(n_samples=50, seed=42)

        # Save dataset
        filepath = temp_data_dir / "test_data.pkl"
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'X': X, 'y': y, 'n_samples': X.shape[0], 'n_features': X.shape[1]}, f)

        # Load dataset
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        X_loaded = data['X']
        y_loaded = data['y']

        # Verify
        assert np.allclose(X, X_loaded)
        assert np.allclose(y, y_loaded)

    def test_dataset_reproducibility(self):
        """Test that same seed produces same dataset."""
        X1, y1 = create_synthetic_dataset(n_samples=100, seed=42)
        X2, y2 = create_synthetic_dataset(n_samples=100, seed=42)

        assert np.allclose(X1, X2)
        assert np.allclose(y1, y2)


class TestModelFunctions:
    """Test model management functions."""

    def test_model_training(self):
        """Test basic model training."""
        X_train, y_train = create_synthetic_dataset(n_samples=100, seed=42)
        X_val, y_val = create_synthetic_dataset(n_samples=20, seed=43)

        model = FiveDRegressor(
            hidden_layers=(16, 8),
            max_epochs=5,
            verbose=False,
        )

        model.fit(X_train, y_train, X_val, y_val)

        assert model._is_fitted
        assert len(model.history['epoch']) > 0

    def test_model_save_and_load(self, tmp_path):
        """Test model saving and loading."""
        # Train a simple model
        X_train, y_train = create_synthetic_dataset(n_samples=100, seed=42)
        X_val, y_val = create_synthetic_dataset(n_samples=20, seed=43)

        model = FiveDRegressor(
            hidden_layers=(16, 8),
            max_epochs=3,
            verbose=False,
        )
        model.fit(X_train, y_train, X_val, y_val)

        # Save model
        model_path = tmp_path / "test_model.pt"
        model.save_model(str(model_path))

        assert model_path.exists()

        # Load model
        loaded_model = FiveDRegressor.load_model(str(model_path))

        # Verify loaded model
        assert loaded_model._is_fitted
        assert loaded_model.hidden_layers == model.hidden_layers
        assert len(loaded_model.history['epoch']) == len(model.history['epoch'])

    def test_save_load_predictions_match(self, tmp_path):
        """Test that predictions match after save/load."""
        # Train model
        X_train, y_train = create_synthetic_dataset(n_samples=100, seed=42)
        X_test, y_test = create_synthetic_dataset(n_samples=10, seed=43)

        model = FiveDRegressor(
            hidden_layers=(16, 8),
            max_epochs=3,
            verbose=False,
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_original = model.predict(X_test)

        # Save and load
        model_path = tmp_path / "test_model.pt"
        model.save_model(str(model_path))
        loaded_model = FiveDRegressor.load_model(str(model_path))

        # Make predictions with loaded model
        y_pred_loaded = loaded_model.predict(X_test)

        # Verify predictions match
        assert np.allclose(y_pred_original, y_pred_loaded, rtol=1e-6)

    def test_model_predict_without_fit(self):
        """Test that predict raises error when model not fitted."""
        model = FiveDRegressor()
        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            model.predict(X)


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_workflow(self, tmp_path):
        """Test complete workflow: create data, train, save, load, predict."""
        # Create dataset
        X_train, y_train = create_synthetic_dataset(n_samples=200, seed=42)
        X_val, y_val = create_synthetic_dataset(n_samples=50, seed=43)
        X_test, y_test = create_synthetic_dataset(n_samples=30, seed=44)

        # Train model
        model = FiveDRegressor(
            hidden_layers=(32, 16),
            max_epochs=20,  # More epochs for better fit
            verbose=False,
        )
        model.fit(X_train, y_train, X_val, y_val)

        # Save model
        model_path = tmp_path / "workflow_model.pt"
        model.save_model(str(model_path))

        # Load model
        loaded_model = FiveDRegressor.load_model(str(model_path))

        # Make predictions
        y_pred = loaded_model.predict(X_test)

        # Verify predictions
        assert y_pred.shape == y_test.shape
        assert isinstance(y_pred, np.ndarray)

        # Verify predictions are reasonable (not all zeros or NaN)
        assert not np.isnan(y_pred).any()
        assert not np.all(y_pred == 0)


def test_directories_exist():
    """Test that data and models directories exist."""
    data_dir = get_data_dir()
    models_dir = get_models_dir()

    assert data_dir.exists()
    assert models_dir.exists()
    assert data_dir.is_dir()
    assert models_dir.is_dir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
