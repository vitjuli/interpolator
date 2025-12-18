import copy
from typing import Optional, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchmetrics
from fvcore.nn import FlopCountAnalysis, parameter_count


class FiveDRegressor(nn.Module):
    """Lightweight neural network for 5D regression with fit/predict API.

    Features:
    - Configurable hidden layers
    - Mini-batch training for speed on CPU
    - Early stopping based on validation loss (restores best weights)
    - Reproducible via fixed random seed
    """

    def __init__(
        self,
        hidden_layers=(64, 32, 16),
        learning_rate=1e-3,
        max_epochs=200,
        batch_size=256,
        weight_decay=0.0,
        patience=20,
        min_delta=1e-6,
        random_state=42,
        verbose=False,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        log_every_n_steps=100,
    ):
        """
        Args:
            hidden_layers: Sequence of hidden layer widths.
            learning_rate: Learning rate for Adam optimizer.
            max_epochs: Maximum training epochs.
            batch_size: Mini-batch size for DataLoader.
            weight_decay: L2 regularization term (Adam weight_decay).
            patience: Early stopping patience (epochs).
            min_delta: Minimum improvement in val loss to reset patience.
            random_state: Seed for reproducibility.
            verbose: If True, prints training progress.
            use_wandb: If True, logs metrics to Weights & Biases.
            wandb_project: Project name for wandb (required if use_wandb=True).
            wandb_run_name: Run name for wandb (optional).
        """
        super().__init__()

        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.use_wandb = bool(use_wandb)
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.hidden_layers = tuple(hidden_layers)
        self.log_every_n_steps = int(log_every_n_steps)

        self.device = torch.device("cpu")  # enforce CPU

        # Reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Build layers dynamically
        layers = []
        in_dim = 5
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, int(h)))
            layers.append(nn.ReLU())
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

        self.to(self.device)
        self._is_fitted = False

        # Initialize TorchMetrics
        self.train_mse_metric = torchmetrics.MeanSquaredError().to(self.device)
        self.val_mse_metric = torchmetrics.MeanSquaredError().to(self.device)
        self.train_r2_metric = torchmetrics.R2Score().to(self.device)
        self.val_r2_metric = torchmetrics.R2Score().to(self.device)
        self.train_mae_metric = torchmetrics.MeanAbsoluteError().to(self.device)
        self.val_mae_metric = torchmetrics.MeanAbsoluteError().to(self.device)

        # Store training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_r2": [],
            "val_r2": [],
            "train_mae": [],
            "val_mae": [],
            "epoch": [],
        }

    def forward(self, x):
        return self.network(x)

    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and total parameters in the model.

        Returns:
            Dictionary with parameter counts and breakdown by layer.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        param_breakdown = {}
        for name, param in self.named_parameters():
            param_breakdown[name] = param.numel()

        return {
            "total": total_params,
            "trainable": trainable_params,
            "breakdown": param_breakdown,
        }

    def count_flops(self, batch_size: int = 1) -> Dict[str, Any]:
        """
        Count FLOPs (floating point operations) for a forward pass.

        Args:
            batch_size: Batch size for FLOP calculation.

        Returns:
            Dictionary with FLOP counts and analysis.
        """
        # Create dummy input
        dummy_input = torch.randn(batch_size, 5).to(self.device)

        # Count FLOPs
        flops = FlopCountAnalysis(self, dummy_input)
        total_flops = flops.total()

        # Calculate FLOPs per layer
        flops_by_operator = flops.by_operator()

        return {
            "total_flops": total_flops,
            "flops_per_sample": total_flops // batch_size,
            "by_operator": {k: int(v) for k, v in flops_by_operator.items()},
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary including architecture and computational cost.

        Returns:
            Dictionary with model architecture, parameters, and FLOPs.
        """
        params = self.count_parameters()
        flops = self.count_flops(batch_size=1)

        return {
            "architecture": {
                "input_dim": 5,
                "hidden_layers": self.hidden_layers,
                "output_dim": 1,
                "total_layers": len(self.hidden_layers) + 1,
            },
            "parameters": params,
            "flops": flops,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "weight_decay": self.weight_decay,
                "patience": self.patience,
            },
        }

    @staticmethod
    def _check_Xy(X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 5:
            raise ValueError(f"Expected X shape (n_samples, 5). Got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")

        if y is None:
            return X, None

        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError(f"Expected y shape (n_samples,). Got {y.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X and y must have same length. Got {X.shape[0]} vs {y.shape[0]}")
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values.")
        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model with optional wandb logging and comprehensive metrics.

        Args:
            X_train: numpy array of shape (n_train, 5)
            y_train: numpy array of shape (n_train,)
            X_val: optional validation features
            y_val: optional validation targets
        """
        # Initialize wandb if requested
        wandb_run = None
        if self.use_wandb:
            try:
                import wandb
                config = {
                    "architecture": "FiveDRegressor",
                    "hidden_layers": self.hidden_layers,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "max_epochs": self.max_epochs,
                    "weight_decay": self.weight_decay,
                    "patience": self.patience,
                }
                # Add model complexity metrics
                model_summary = self.get_model_summary()
                config.update({
                    "total_parameters": model_summary["parameters"]["total"],
                    "total_flops": model_summary["flops"]["total_flops"],
                })

                wandb_run = wandb.init(
                    project=self.wandb_project or "fivedreg-training",
                    name=self.wandb_run_name,
                    config=config,
                    reinit=True,
                )
                wandb.watch(self, log="all", log_freq=10)
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                self.use_wandb = False

        X_train, y_train = self._check_Xy(X_train, y_train)
        use_val = X_val is not None and y_val is not None
        if use_val:
            X_val, y_val = self._check_Xy(X_val, y_val)

        # Tensors + DataLoader for speed
        train_ds = TensorDataset(
            torch.from_numpy(X_train).to(self.device),
            torch.from_numpy(y_train).to(self.device),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if use_val:
            X_val_t = torch.from_numpy(X_val).to(self.device)
            y_val_t = torch.from_numpy(y_val).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_val = float("inf")
        best_state = None
        patience_left = self.patience

        # Reset history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_r2": [],
            "val_r2": [],
            "train_mae": [],
            "val_mae": [],
            "epoch": [],
        }

        # Global step counter for step-level logging
        global_step = 0

        for epoch in range(self.max_epochs):
            self.train()
            running = 0.0
            n = 0

            # Reset metrics
            self.train_mse_metric.reset()
            self.train_r2_metric.reset()
            self.train_mae_metric.reset()

            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self(xb).squeeze(1)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * xb.shape[0]
                n += xb.shape[0]

                # Update TorchMetrics
                self.train_mse_metric.update(pred, yb)
                self.train_r2_metric.update(pred, yb)
                self.train_mae_metric.update(pred, yb)

                global_step += 1

                # Step-level logging (every N steps)
                if self.use_wandb and wandb_run is not None and global_step % self.log_every_n_steps == 0:
                    import wandb
                    wandb.log({
                        "step/train_loss": float(loss.item()),
                        "step/train_mse": float(self.train_mse_metric.compute().item()),
                        "step/global_step": global_step,
                        "step/epoch": epoch + 1,
                    }, step=global_step)

            train_loss = running / max(n, 1)
            train_r2 = float(self.train_r2_metric.compute().item())
            train_mae = float(self.train_mae_metric.compute().item())

            # Store in history
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["train_r2"].append(train_r2)
            self.history["train_mae"].append(train_mae)

            # Validation
            if use_val:
                self.eval()
                self.val_mse_metric.reset()
                self.val_r2_metric.reset()
                self.val_mae_metric.reset()

                with torch.no_grad():
                    val_pred = self(X_val_t).squeeze(1)
                    val_loss = float(criterion(val_pred, y_val_t).item())

                    # Compute validation metrics with TorchMetrics
                    self.val_mse_metric.update(val_pred, y_val_t)
                    self.val_r2_metric.update(val_pred, y_val_t)
                    self.val_mae_metric.update(val_pred, y_val_t)

                val_r2 = float(self.val_r2_metric.compute().item())
                val_mae = float(self.val_mae_metric.compute().item())

                # Store in history
                self.history["val_loss"].append(val_loss)
                self.history["val_r2"].append(val_r2)
                self.history["val_mae"].append(val_mae)

                # Log to wandb
                if self.use_wandb and wandb_run is not None:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/r2": train_r2,
                        "train/mae": train_mae,
                        "val/loss": val_loss,
                        "val/r2": val_r2,
                        "val/mae": val_mae,
                        "early_stopping/patience_left": patience_left,
                        "early_stopping/best_val_loss": best_val,
                    })

                improved = (best_val - val_loss) > self.min_delta
                if improved:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.state_dict())
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

                if self.verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                    print(f"Epoch {epoch+1:4d}: train_mse={train_loss:.6f} val_mse={val_loss:.6f} "
                          f"train_r2={train_r2:.4f} val_r2={val_r2:.4f}")
            else:
                # No validation set
                if self.use_wandb and wandb_run is not None:
                    wandb.log({
                        "epoch": epoch + 1,
                        "train/loss": train_loss,
                        "train/r2": train_r2,
                        "train/mae": train_mae,
                    })

                if self.verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                    print(f"Epoch {epoch+1:4d}: train_mse={train_loss:.6f} train_r2={train_r2:.4f}")

        if use_val and best_state is not None:
            self.load_state_dict(best_state)

        # Finish wandb run
        if self.use_wandb and wandb_run is not None:
            wandb.finish()

        self._is_fitted = True
        return self

    def predict(self, X):
        """
        Predict targets.

        Args:
            X: numpy array of shape (n_samples, 5)

        Returns:
            numpy array of shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        X, _ = self._check_Xy(X, None)
        self.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X).to(self.device)
            pred = self(Xt).squeeze(1).cpu().numpy()
        return pred.astype(np.float32, copy=False)

    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save the model (e.g., 'model.pt')
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'hyperparameters': {
                'hidden_layers': self.hidden_layers,
                'learning_rate': self.learning_rate,
                'max_epochs': self.max_epochs,
                'batch_size': self.batch_size,
                'weight_decay': self.weight_decay,
                'patience': self.patience,
                'min_delta': self.min_delta,
                'random_state': self.random_state,
                'verbose': self.verbose,
                'use_wandb': self.use_wandb,
                'wandb_project': self.wandb_project,
                'wandb_run_name': self.wandb_run_name,
                'log_every_n_steps': self.log_every_n_steps,
            },
            'history': self.history,
            'is_fitted': self._is_fitted,
        }
        torch.save(save_dict, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> 'FiveDRegressor':
        """
        Load model from disk.

        Args:
            filepath: Path to the saved model (e.g., 'model.pt')

        Returns:
            Loaded FiveDRegressor instance
        """
        checkpoint = torch.load(filepath, map_location='cpu')

        # Create new instance with saved hyperparameters
        model = cls(**checkpoint['hyperparameters'])

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Restore training history and fitted state
        model.history = checkpoint['history']
        model._is_fitted = checkpoint['is_fitted']

        return model