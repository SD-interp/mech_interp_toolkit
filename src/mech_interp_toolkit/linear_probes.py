import torch
from .activations import ActivationDict
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from typing import Optional, Literal
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


class LinearProbe:
    def __init__(
        self,
        target_type: Literal["classification", "regression"],
        broadcast_target: bool = True,
        test_split: float = 0.2,
        **kwargs,
    ):
        self.target_type = target_type
        self.broadcast_target = broadcast_target

        if test_split <= 0.0 or test_split >= 1.0:
            raise ValueError("test_split must be between 0.0 and 1.0")
        self.test_split = test_split

        if target_type == "classification":
            self.linear_model = LogisticRegression(**kwargs)
        elif target_type == "regression":
            self.linear_model = LinearRegression(**kwargs)
        else:
            raise ValueError("target_type must be 'classification' or 'regression'")

        self.weight: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray | float] = None

    def _process_batch(
        self, inputs: np.ndarray, target: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Helper to flatten/broadcast a specific batch (Train or Test)."""
        if inputs.ndim == 2:
            # Shape: (Batch, D_model) -> Treat as single position
            positions = 1
        elif inputs.ndim == 3:
            # Shape: (Batch, Pos, D_model)
            positions = inputs.shape[1]
        else:
            raise ValueError(f"Unexpected input shape: {inputs.shape}")

        d_model = inputs.shape[-1]

        inputs_flat = inputs.reshape(-1, d_model)

        target_flat = None
        if target is not None:
            if self.broadcast_target and positions > 1:
                if target.ndim != 1:
                    raise ValueError(
                        f"broadcast_target=True expects 1D targets (batch_size,), "
                        f"but got shape {target.shape}. If you have token-level targets, "
                        "set broadcast_target=False."
                    )
                target_flat = np.repeat(target, positions, axis=0)
            else:
                target_flat = target.reshape(-1) if target.ndim > 1 else target

        return inputs_flat, target_flat

    def prepare_data(
        self, activations: ActivationDict, target: torch.Tensor | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if len(activations) != 1:
            raise ValueError("Only single components are supported")

        # Raw inputs: (Batch, [Pos], D_model)
        inputs_full = list(activations.values())[0].cpu().numpy()

        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        indices = np.arange(inputs_full.shape[0])
        train_idx, test_idx = train_test_split(indices, test_size=self.test_split)

        # Slice data based on indices
        X_train_raw = inputs_full[train_idx]
        y_train_raw = target[train_idx]
        X_test_raw = inputs_full[test_idx]
        y_test_raw = target[test_idx]

        # Now flatten/broadcast train and test independently
        X_train, y_train = self._process_batch(X_train_raw, y_train_raw)
        X_test, y_test = self._process_batch(X_test_raw, y_test_raw)

        return X_train, X_test, y_train, y_test

    def display_metrics(self, pred: np.ndarray, y: np.ndarray, label: str):
        if self.target_type == "classification":
            metric_name = "Accuracy"
            metric = accuracy_score(y, pred)
        elif self.target_type == "regression":
            metric_name = "RMSE"
            metric = root_mean_squared_error(y, pred)

        print(f"{label} {metric_name}: {metric:.4f}")

    def fit(self, activations: ActivationDict, target: torch.Tensor | np.ndarray):
        X_train, X_test, y_train, y_test = self.prepare_data(activations, target)

        if y_test is None or y_train is None:
            raise ValueError("Target cannot be None for fitting the linear probe.")
        
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

        self.linear_model.fit(X_train, y_train)
        self.weight = self.linear_model.coef_
        self.bias = self.linear_model.intercept_

        pred_train = self.linear_model.predict(X_train)
        pred_test = self.linear_model.predict(X_test)

        self.display_metrics(pred_train, y_train, label="Train")
        self.display_metrics(pred_test, y_test, label="Test")

        return self

    def predict(
        self,
        activations: ActivationDict,
        target: Optional[torch.Tensor | np.ndarray] = None,
        label="Inference",
    ) -> np.ndarray:
        if self.weight is None:  # Simple check
            raise ValueError("The linear probe has not been fitted yet.")

        inputs_full = list(activations.values())[0].cpu().numpy()

        if target is not None:
            if isinstance(target, torch.Tensor):
                target = target.cpu().numpy()

        # Process the entire batch for inference (no splitting needed here)
        inputs, target = self._process_batch(inputs_full, target)

        preds = self.linear_model.predict(inputs)
        print(f"{label} set size: {len(inputs)}")

        if target is not None:
            self.display_metrics(preds, target, label=label)

        return preds