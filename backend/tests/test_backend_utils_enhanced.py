"""
Enhanced tests for backend utility functions to improve coverage.

Tests the utility wrapper functions that weren't covered:
- save_dataset() and load_dataset()
- list_datasets()
- save_model() and load_model() wrapper functions
- list_models()
- create_train_val_test_datasets()
- Error handling (FileNotFoundError, etc.)
"""

import pytest
import numpy as np
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
    create_train_val_test_datasets,
    get_data_dir,
    get_models_dir,
)


class TestSaveLoadDataset:
    """Test save_dataset and load_dataset utility functions."""

    def test_save_dataset_basic(self):
        """Test saving dataset using save_dataset() function."""
        X, y = create_synthetic_dataset(n_samples=50, seed=42)

        # Save using the utility function
        filepath = save_dataset(X, y, "test_save_basic.pkl")

        assert filepath.exists()
        assert filepath.name == "test_save_basic.pkl"

        # Clean up
        filepath.unlink()

    def test_save_dataset_auto_extension(self):
        """Test that save_dataset automatically adds .pkl extension."""
        X, y = create_synthetic_dataset(n_samples=30, seed=42)

        # Save without .pkl extension
        filepath = save_dataset(X, y, "test_no_ext")

        assert filepath.exists()
        assert filepath.name == "test_no_ext.pkl"

        # Clean up
        filepath.unlink()

    def test_load_dataset_basic(self):
        """Test loading dataset using load_dataset() function."""
        X_orig, y_orig = create_synthetic_dataset(n_samples=40, seed=42)

        # Save first
        save_dataset(X_orig, y_orig, "test_load_basic.pkl")

        # Load using utility function
        X_loaded, y_loaded = load_dataset("test_load_basic.pkl")

        assert np.allclose(X_orig, X_loaded)
        assert np.allclose(y_orig, y_loaded)

        # Clean up
        (get_data_dir() / "test_load_basic.pkl").unlink()

    def test_load_dataset_auto_extension(self):
        """Test that load_dataset automatically adds .pkl extension."""
        X_orig, y_orig = create_synthetic_dataset(n_samples=25, seed=42)

        # Save first
        save_dataset(X_orig, y_orig, "test_auto_ext.pkl")

        # Load without .pkl extension
        X_loaded, y_loaded = load_dataset("test_auto_ext")

        assert np.allclose(X_orig, X_loaded)
        assert np.allclose(y_orig, y_loaded)

        # Clean up
        (get_data_dir() / "test_auto_ext.pkl").unlink()

    def test_load_dataset_file_not_found(self):
        """Test that load_dataset raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_dataset("nonexistent_dataset.pkl")

    def test_list_datasets(self):
        """Test list_datasets() function."""
        # Save a couple of datasets
        X1, y1 = create_synthetic_dataset(n_samples=10, seed=42)
        X2, y2 = create_synthetic_dataset(n_samples=20, seed=43)

        save_dataset(X1, y1, "test_list_1.pkl")
        save_dataset(X2, y2, "test_list_2.pkl")

        # List datasets
        datasets = list_datasets()

        assert "test_list_1.pkl" in datasets
        assert "test_list_2.pkl" in datasets
        assert isinstance(datasets, list)

        # Clean up
        (get_data_dir() / "test_list_1.pkl").unlink()
        (get_data_dir() / "test_list_2.pkl").unlink()


class TestSaveLoadModel:
    """Test save_model and load_model utility wrapper functions."""

    def test_save_model_basic(self):
        """Test saving model using save_model() wrapper function."""
        # Train a simple model
        X_train, y_train = create_synthetic_dataset(n_samples=100, seed=42)

        model = FiveDRegressor(
            hidden_layers=(16, 8),
            max_epochs=3,
            verbose=False,
        )
        model.fit(X_train, y_train)

        # Save using wrapper function
        filepath = save_model(model, "test_save_model_basic.pt")

        assert filepath.exists()
        assert filepath.name == "test_save_model_basic.pt"

        # Clean up
        filepath.unlink()

    def test_save_model_auto_extension(self):
        """Test that save_model automatically adds .pt extension."""
        # Train a simple model
        X_train, y_train = create_synthetic_dataset(n_samples=100, seed=42)

        model = FiveDRegressor(
            hidden_layers=(16, 8),
            max_epochs=3,
            verbose=False,
        )
        model.fit(X_train, y_train)

        # Save without .pt extension
        filepath = save_model(model, "test_no_ext_model")

        assert filepath.exists()
        assert filepath.name == "test_no_ext_model.pt"

        # Clean up
        filepath.unlink()

    def test_load_model_basic(self):
        """Test loading model using load_model() wrapper function."""
        # Train and save a model
        X_train, y_train = create_synthetic_dataset(n_samples=100, seed=42)

        model = FiveDRegressor(
            hidden_layers=(16, 8),
            max_epochs=3,
            verbose=False,
        )
        model.fit(X_train, y_train)

        save_model(model, "test_load_model_basic.pt")

        # Load using wrapper function
        loaded_model = load_model("test_load_model_basic.pt")

        assert loaded_model._is_fitted
        assert loaded_model.hidden_layers == (16, 8)

        # Clean up
        (get_models_dir() / "test_load_model_basic.pt").unlink()

    def test_load_model_auto_extension(self):
        """Test that load_model automatically adds .pt extension."""
        # Train and save a model
        X_train, y_train = create_synthetic_dataset(n_samples=100, seed=42)

        model = FiveDRegressor(
            hidden_layers=(16, 8),
            max_epochs=3,
            verbose=False,
        )
        model.fit(X_train, y_train)

        save_model(model, "test_auto_ext_model.pt")

        # Load without .pt extension
        loaded_model = load_model("test_auto_ext_model")

        assert loaded_model._is_fitted
        assert loaded_model.hidden_layers == (16, 8)

        # Clean up
        (get_models_dir() / "test_auto_ext_model.pt").unlink()

    def test_load_model_file_not_found(self):
        """Test that load_model raises FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model("nonexistent_model.pt")

    def test_list_models(self):
        """Test list_models() function."""
        # Train and save a couple of models
        X_train, y_train = create_synthetic_dataset(n_samples=100, seed=42)

        model1 = FiveDRegressor(hidden_layers=(16,), max_epochs=2, verbose=False)
        model1.fit(X_train, y_train)
        save_model(model1, "test_list_model_1.pt")

        model2 = FiveDRegressor(hidden_layers=(32,), max_epochs=2, verbose=False)
        model2.fit(X_train, y_train)
        save_model(model2, "test_list_model_2.pt")

        # List models
        models = list_models()

        assert "test_list_model_1.pt" in models
        assert "test_list_model_2.pt" in models
        assert isinstance(models, list)

        # Clean up
        (get_models_dir() / "test_list_model_1.pt").unlink()
        (get_models_dir() / "test_list_model_2.pt").unlink()

    def test_save_load_predictions_consistency(self):
        """Test that predictions are consistent after save/load using utility functions."""
        # Train model
        X_train, y_train = create_synthetic_dataset(n_samples=150, seed=42)
        X_test, _ = create_synthetic_dataset(n_samples=20, seed=99)

        model = FiveDRegressor(
            hidden_layers=(32, 16),
            max_epochs=5,
            verbose=False,
        )
        model.fit(X_train, y_train)

        # Predictions before save
        y_pred_before = model.predict(X_test)

        # Save and load using utility functions
        save_model(model, "test_consistency.pt")
        loaded_model = load_model("test_consistency.pt")

        # Predictions after load
        y_pred_after = loaded_model.predict(X_test)

        # Should be identical
        assert np.allclose(y_pred_before, y_pred_after, rtol=1e-6)

        # Clean up
        (get_models_dir() / "test_consistency.pt").unlink()


class TestCreateTrainValTest:
    """Test create_train_val_test_datasets() convenience function."""

    def test_create_train_val_test_no_save(self):
        """Test creating train/val/test splits without saving."""
        X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_datasets(
            n_train=100,
            n_val=20,
            n_test=30,
            seed=42,
            save=False
        )

        # Check shapes
        assert X_train.shape == (100, 5)
        assert y_train.shape == (100,)
        assert X_val.shape == (20, 5)
        assert y_val.shape == (20,)
        assert X_test.shape == (30, 5)
        assert y_test.shape == (30,)

    def test_create_train_val_test_with_save(self):
        """Test creating train/val/test splits with saving."""
        X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_datasets(
            n_train=80,
            n_val=15,
            n_test=25,
            seed=42,
            save=True
        )

        # Check that files were created
        data_dir = get_data_dir()
        assert (data_dir / "train_data.pkl").exists()
        assert (data_dir / "val_data.pkl").exists()
        assert (data_dir / "test_data.pkl").exists()

        # Load and verify
        X_train_loaded, y_train_loaded = load_dataset("train_data.pkl")
        assert np.allclose(X_train, X_train_loaded)
        assert np.allclose(y_train, y_train_loaded)

        # Clean up
        (data_dir / "train_data.pkl").unlink()
        (data_dir / "val_data.pkl").unlink()
        (data_dir / "test_data.pkl").unlink()

    def test_create_train_val_test_reproducibility(self):
        """Test that same seed produces same splits."""
        result1 = create_train_val_test_datasets(
            n_train=50, n_val=10, n_test=15, seed=123, save=False
        )
        result2 = create_train_val_test_datasets(
            n_train=50, n_val=10, n_test=15, seed=123, save=False
        )

        # All arrays should be identical
        for arr1, arr2 in zip(result1, result2):
            assert np.allclose(arr1, arr2)


class TestCreateSyntheticWithSavePath:
    """Test create_synthetic_dataset with save_path parameter."""

    def test_create_synthetic_with_save_path(self):
        """Test that create_synthetic_dataset can save directly."""
        X, y = create_synthetic_dataset(
            n_samples=60,
            seed=42,
            save_path="test_direct_save.pkl"
        )

        # Check that file was created
        filepath = get_data_dir() / "test_direct_save.pkl"
        assert filepath.exists()

        # Load and verify
        X_loaded, y_loaded = load_dataset("test_direct_save.pkl")
        assert np.allclose(X, X_loaded)
        assert np.allclose(y, y_loaded)

        # Clean up
        filepath.unlink()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_dataset_creates_data_in_dict(self):
        """Test that saved dataset has correct dictionary structure."""
        X, y = create_synthetic_dataset(n_samples=15, seed=42)
        filepath = save_dataset(X, y, "test_dict_structure.pkl")

        # Load manually to check structure
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        assert 'X' in data
        assert 'y' in data
        assert 'n_samples' in data
        assert 'n_features' in data
        assert data['n_samples'] == 15
        assert data['n_features'] == 5

        # Clean up
        filepath.unlink()

    def test_empty_list_datasets_returns_list(self):
        """Test that list_datasets() returns empty list when no datasets."""
        # Clean up data directory temporarily
        data_dir = get_data_dir()
        existing_files = list(data_dir.glob("*.pkl"))

        # Move files temporarily if any exist
        temp_files = []
        for f in existing_files:
            temp_path = data_dir / f"_temp_{f.name}"
            f.rename(temp_path)
            temp_files.append((f, temp_path))

        # Test
        datasets = list_datasets()
        assert isinstance(datasets, list)

        # Restore files
        for original_path, temp_path in temp_files:
            temp_path.rename(original_path)

    def test_empty_list_models_returns_list(self):
        """Test that list_models() returns empty list when no models."""
        # Clean up models directory temporarily
        models_dir = get_models_dir()
        existing_files = list(models_dir.glob("*.pt"))

        # Move files temporarily if any exist
        temp_files = []
        for f in existing_files:
            temp_path = models_dir / f"_temp_{f.name}"
            f.rename(temp_path)
            temp_files.append((f, temp_path))

        # Test
        models = list_models()
        assert isinstance(models, list)

        # Restore files
        for original_path, temp_path in temp_files:
            temp_path.rename(original_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
