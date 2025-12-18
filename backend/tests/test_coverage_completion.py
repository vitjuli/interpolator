"""
Additional tests to achieve 100% code coverage.

This file covers:
1. Error validation paths in data.py
2. Error validation paths in model.py _check_Xy
3. Model introspection methods (count_parameters, count_flops, get_model_summary)
4. Wandb integration code paths
5. Verbose training without validation
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys

from fivedreg import FiveDRegressor, create_synthetic_dataset
from fivedreg.data import load_dataset


class TestDataValidationErrors:
    """Test error conditions in data.py load_dataset function."""

    def test_X_not_2d_error(self, tmp_path):
        """Test that 1D X raises ValueError."""
        import pickle

        # Create dataset with 1D X
        filepath = tmp_path / "bad_1d.pkl"
        bad_data = {
            'X': np.array([1, 2, 3, 4, 5]),  # 1D instead of 2D
            'y': np.array([1, 2, 3, 4, 5])
        }
        with open(filepath, 'wb') as f:
            pickle.dump(bad_data, f)

        # Should raise error about X being 2D
        with pytest.raises(ValueError, match="X must be 2D"):
            load_dataset(str(filepath))

    def test_X_wrong_feature_count_error(self, tmp_path):
        """Test that X with != 5 features raises ValueError."""
        import pickle

        # Create dataset with wrong number of features
        filepath = tmp_path / "bad_features.pkl"
        bad_data = {
            'X': np.random.randn(10, 3),  # 3 features instead of 5
            'y': np.random.randn(10)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(bad_data, f)

        # Should raise error about feature count
        with pytest.raises(ValueError, match="Expected 5 features"):
            load_dataset(str(filepath))

    def test_X_contains_inf_error(self, tmp_path):
        """Test that X with inf values raises ValueError."""
        import pickle

        # Create dataset with inf values in X
        filepath = tmp_path / "inf_X.pkl"
        X = np.random.randn(10, 5)
        X[2, 3] = np.inf  # Add inf value
        bad_data = {
            'X': X,
            'y': np.random.randn(10)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(bad_data, f)

        # Should raise error about inf values
        with pytest.raises(ValueError, match="X contains infinite values"):
            load_dataset(str(filepath))

    def test_y_2d_reshape_to_1d(self, tmp_path):
        """Test that 2D y with shape (n, 1) is reshaped to 1D."""
        import pickle

        # Create dataset with 2D y
        filepath = tmp_path / "y_2d.pkl"
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)  # 2D with shape (10, 1)
        data = {
            'X': X,
            'y': y
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        # Load and verify y is reshaped to 1D
        X_loaded, y_loaded = load_dataset(str(filepath))

        assert y_loaded.ndim == 1
        assert y_loaded.shape == (10,)
        assert np.allclose(y_loaded, y.reshape(-1))

    def test_y_wrong_ndim_error(self, tmp_path):
        """Test that y with wrong dimensions raises ValueError."""
        import pickle

        # Create dataset with 2D y that can't be reshaped
        filepath = tmp_path / "bad_y.pkl"
        bad_data = {
            'X': np.random.randn(10, 5),
            'y': np.random.randn(10, 3)  # 2D with multiple columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(bad_data, f)

        # Should raise error about y dimensions
        with pytest.raises(ValueError, match="y must be 1D"):
            load_dataset(str(filepath))

    def test_X_y_length_mismatch_error(self, tmp_path):
        """Test that X and y with different lengths raise ValueError."""
        import pickle

        # Create dataset with mismatched lengths
        filepath = tmp_path / "mismatch.pkl"
        bad_data = {
            'X': np.random.randn(10, 5),
            'y': np.random.randn(15)  # Different length
        }
        with open(filepath, 'wb') as f:
            pickle.dump(bad_data, f)

        # Should raise error about length mismatch
        with pytest.raises(ValueError, match="same number of samples"):
            load_dataset(str(filepath))

    def test_y_contains_inf_error(self, tmp_path):
        """Test that y with inf values raises ValueError."""
        import pickle

        # Create dataset with inf values in y
        filepath = tmp_path / "inf_y.pkl"
        y = np.random.randn(10)
        y[5] = np.inf  # Add inf value
        bad_data = {
            'X': np.random.randn(10, 5),
            'y': y
        }
        with open(filepath, 'wb') as f:
            pickle.dump(bad_data, f)

        # Should raise error about inf values
        with pytest.raises(ValueError, match="y contains infinite values"):
            load_dataset(str(filepath))


class TestModelCheckXyErrors:
    """Test error conditions in FiveDRegressor._check_Xy method."""

    def test_y_2d_reshape_in_check_xy(self):
        """Test that _check_Xy reshapes 2D y to 1D."""
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10, 1).astype(np.float32)  # 2D

        X_checked, y_checked = FiveDRegressor._check_Xy(X, y)

        assert y_checked.ndim == 1
        assert y_checked.shape == (10,)

    def test_y_wrong_ndim_in_check_xy(self):
        """Test that _check_Xy raises error for wrong y dimensions."""
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)  # 2D with multiple columns

        with pytest.raises(ValueError, match="Expected y shape"):
            FiveDRegressor._check_Xy(X, y)

    def test_y_contains_inf_in_check_xy(self):
        """Test that _check_Xy raises error for inf in y."""
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)
        y[3] = np.inf

        with pytest.raises(ValueError, match="y contains NaN or infinite values"):
            FiveDRegressor._check_Xy(X, y)


class TestModelIntrospection:
    """Test model introspection methods."""

    def test_count_parameters(self):
        """Test count_parameters method."""
        model = FiveDRegressor(hidden_layers=(32, 16), verbose=False)

        params = model.count_parameters()

        assert "total" in params
        assert "trainable" in params
        assert "breakdown" in params
        assert params["total"] > 0
        assert params["trainable"] == params["total"]  # All params should be trainable
        assert isinstance(params["breakdown"], dict)

    def test_count_flops(self):
        """Test count_flops method."""
        model = FiveDRegressor(hidden_layers=(32, 16), verbose=False)

        flops = model.count_flops(batch_size=1)

        assert "total_flops" in flops
        assert "flops_per_sample" in flops
        assert "by_operator" in flops
        assert flops["total_flops"] > 0

    def test_count_flops_with_batch(self):
        """Test count_flops with larger batch size."""
        model = FiveDRegressor(hidden_layers=(16, 8), verbose=False)

        flops = model.count_flops(batch_size=32)

        assert flops["total_flops"] > 0
        assert flops["flops_per_sample"] > 0

    def test_get_model_summary(self):
        """Test get_model_summary method."""
        model = FiveDRegressor(
            hidden_layers=(64, 32),
            learning_rate=0.001,
            batch_size=128,
            verbose=False
        )

        summary = model.get_model_summary()

        # Check architecture section
        assert "architecture" in summary
        assert summary["architecture"]["input_dim"] == 5
        assert summary["architecture"]["hidden_layers"] == (64, 32)
        assert summary["architecture"]["output_dim"] == 1

        # Check parameters section
        assert "parameters" in summary
        assert "total" in summary["parameters"]

        # Check flops section
        assert "flops" in summary

        # Check hyperparameters section
        assert "hyperparameters" in summary
        assert summary["hyperparameters"]["learning_rate"] == 0.001
        assert summary["hyperparameters"]["batch_size"] == 128


class TestWandbIntegration:
    """Test wandb integration code paths."""

    def test_training_with_wandb_enabled_mocked(self):
        """Test training with wandb enabled (mocked)."""
        # Mock wandb module
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            X_train, y_train = create_synthetic_dataset(100, seed=42)
            X_val, y_val = create_synthetic_dataset(20, seed=43)

            model = FiveDRegressor(
                hidden_layers=(16, 8),
                max_epochs=3,
                use_wandb=True,
                wandb_project="test_project",
                verbose=False
            )

            model.fit(X_train, y_train, X_val, y_val)

            # Verify wandb was called
            assert mock_wandb.init.called
            assert mock_wandb.finish.called

    def test_training_with_wandb_no_validation_mocked(self):
        """Test training with wandb enabled but no validation set."""
        # Mock wandb module
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            X_train, y_train = create_synthetic_dataset(100, seed=42)

            model = FiveDRegressor(
                hidden_layers=(16, 8),
                max_epochs=3,
                use_wandb=True,
                log_every_n_steps=10,
                verbose=False
            )

            model.fit(X_train, y_train)

            # Verify wandb was called
            assert mock_wandb.init.called
            assert mock_wandb.log.called  # Epoch logging without validation
            assert mock_wandb.finish.called

    def test_wandb_import_failure_graceful(self):
        """Test that wandb import failure is handled gracefully."""
        # Mock wandb to raise ImportError
        def raise_import_error(*args, **kwargs):
            raise ImportError("wandb not installed")

        mock_wandb = MagicMock()
        mock_wandb.init.side_effect = raise_import_error

        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            X_train, y_train = create_synthetic_dataset(50, seed=42)

            model = FiveDRegressor(
                hidden_layers=(8,),
                max_epochs=2,
                use_wandb=True,
                verbose=False
            )

            # Should not raise error, just disable wandb
            model.fit(X_train, y_train)

            assert model._is_fitted


class TestVerboseMode:
    """Test verbose training output."""

    def test_verbose_training_without_validation(self, capsys):
        """Test verbose mode without validation set."""
        X_train, y_train = create_synthetic_dataset(100, seed=42)

        model = FiveDRegressor(
            hidden_layers=(16, 8),
            max_epochs=21,  # To trigger verbose output at epoch 1 and 21
            verbose=True
        )

        model.fit(X_train, y_train)

        # Capture output
        captured = capsys.readouterr()

        # Should have printed training info
        assert "Epoch" in captured.out
        assert "train_mse" in captured.out
        assert "train_r2" in captured.out

    def test_verbose_training_with_validation(self, capsys):
        """Test verbose mode with validation set."""
        X_train, y_train = create_synthetic_dataset(100, seed=42)
        X_val, y_val = create_synthetic_dataset(20, seed=43)

        model = FiveDRegressor(
            hidden_layers=(16, 8),
            max_epochs=21,  # To trigger verbose output
            verbose=True
        )

        model.fit(X_train, y_train, X_val, y_val)

        # Capture output
        captured = capsys.readouterr()

        # Should have printed training and validation info
        assert "Epoch" in captured.out
        assert "train_mse" in captured.out
        assert "val_mse" in captured.out
        assert "val_r2" in captured.out


class TestEdgeCasesCompletion:
    """Additional edge cases for complete coverage."""

    def test_training_triggers_step_logging(self):
        """Test that training with many steps triggers step-level logging."""
        # Mock wandb module
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with patch.dict('sys.modules', {'wandb': mock_wandb}):
            # Use larger dataset and small batch size to generate many steps
            X_train, y_train = create_synthetic_dataset(200, seed=42)
            X_val, y_val = create_synthetic_dataset(40, seed=43)

            model = FiveDRegressor(
                hidden_layers=(8,),
                max_epochs=5,
                batch_size=16,  # Small batch size for more steps
                use_wandb=True,
                log_every_n_steps=5,  # Log every 5 steps
                verbose=False
            )

            model.fit(X_train, y_train, X_val, y_val)

            # Verify step-level logging was called
            log_calls = [call for call in mock_wandb.log.call_args_list
                        if call[0][0].get('step/train_loss') is not None]
            assert len(log_calls) > 0, "Step-level logging should have been called"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
