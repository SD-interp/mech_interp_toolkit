from typing import Literal, Optional, Self

import einops
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from .activation_dict import ActivationDict, LayerComponent


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

        self.location: Optional[LayerComponent] = None

    def _process_batch(
        self, inputs: np.ndarray, target: Optional[np.ndarray], mask: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Helper to flatten/broadcast a specific batch (Train or Test)."""

        positions = inputs.shape[1]
        d_model = inputs.shape[-1]

        # (n_set * pos, d_model)
        inputs_flat = einops.rearrange(inputs, "n_set pos d_model -> (n_set pos) d_model")
        target_flat = None

        if target is not None:
            if target.ndim == 1:
                if self.broadcast_target:
                    target_flat = einops.repeat(target, "n_set -> (n_set pos)", pos=positions)
            elif target.ndim == 2:
                target_flat = einops.rearrange(target, "n_set pos -> (n_set pos)")
            else:
                raise ValueError(f"Incorrect shape {target.shape} for target")

        # Apply attention mask if provided
        if mask is not None:
            # Flatten mask to match inputs_flat shape
            mask_flat = einops.rearrange(mask, "n_set pos -> (n_set pos)")
            if mask_flat.shape[0] != inputs_flat.shape[0]:
                raise ValueError("mask-input shape mismatch")

            inputs_flat = inputs_flat[mask_flat, :]
            if target_flat is not None:
                target_flat = target_flat[mask_flat]

        return inputs_flat, target_flat

    def prepare_data(
        self, activations: ActivationDict, target: torch.Tensor | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if len(activations) != 1:
            raise ValueError("Only single components are supported")

        self.location = list(activations.keys())[0]

        # Raw inputs: (batch, pos, d_model)
        inputs_full = list(activations.values())[0].cpu().detach().numpy()

        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        # Get attention mask if available
        attention_mask = None
        if activations.attention_mask is not None:
            # Attention mask: (batch, pos)
            attention_mask = activations.attention_mask.cpu().numpy()

        # Flatten and apply mask BEFORE splitting to ensure proper stratification
        inputs_flat, target_flat = self._process_batch(inputs_full, target, attention_mask)

        if target_flat is None:
            raise ValueError("Target cannot be None for preparing data.")

        # Indices at token level (after masking)
        indices = np.arange(inputs_flat.shape[0])

        # Use stratified split for classification to ensure both sets have all classes
        stratify = target_flat if self.target_type == "classification" else None

        train_idx, test_idx = train_test_split(
            indices, test_size=self.test_split, stratify=stratify
        )

        # Convert to proper numpy arrays for indexing
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        # Slice flattened data based on token indices
        X_train = inputs_flat[train_idx, :]  # noqa: N806
        y_train = target_flat[train_idx]
        X_test = inputs_flat[test_idx, :]  # noqa: N806
        y_test = target_flat[test_idx]

        return X_train, X_test, y_train, y_test

    def display_metrics(self, pred: np.ndarray, y: np.ndarray, label: str) -> None:
        if self.target_type == "classification":
            metric_name = "Accuracy"
            metric = accuracy_score(y, pred)
        elif self.target_type == "regression":
            metric_name = "RMSE"
            metric = root_mean_squared_error(y, pred)

        print(f"{label} {metric_name}: {metric:.4f}")

    def fit(self, activations: ActivationDict, target: torch.Tensor | np.ndarray) -> Self:
        X_train, X_test, y_train, y_test = self.prepare_data(activations, target)  # noqa: N806

        if y_test is None or y_train is None:
            raise ValueError("Target cannot be None for fitting the linear probe.")

        self.location = list(activations.keys())[0]

        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        print(f"y_train unique values: {np.unique(y_train)}, counts: {np.bincount(y_train.astype(int)) if self.target_type == 'classification' else 'N/A'}")
        print(f"y_test unique values: {np.unique(y_test)}, counts: {np.bincount(y_test.astype(int)) if self.target_type == 'classification' else 'N/A'}")

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

        # Get attention mask if available
        mask = None
        if activations.attention_mask is not None:
            mask = activations.attention_mask.cpu().numpy()

        # Process the entire batch for inference (no splitting needed here)
        inputs, target = self._process_batch(inputs_full, target, mask)

        preds = self.linear_model.predict(inputs)
        print(f"{label} set size: {len(inputs)}")

        if target is not None:
            self.display_metrics(preds, target, label=label)

        return preds
