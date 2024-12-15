import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.tree import DecisionTreeRegressor

from .utils import ConvergenceHistory


class RandomForestMSE:
    def __init__(
        self, n_estimators: int, tree_params: dict[str, Any] | None = None
    ) -> None:
        """
        Handmade random forest regressor.

        Classic ML algorithm that trains a set of independent tall decision trees and averages its predictions. Employs scikit-learn `DecisionTreeRegressor` under the hood.

        Args:
            n_estimators (int): Number of trees in the forest.
            tree_params (dict[str, Any] | None, optional): Parameters for sklearn trees. Defaults to None.
        """
        self.n_estimators = n_estimators
        if tree_params is None:
            tree_params = {}
        self.forest = [
            DecisionTreeRegressor(**tree_params) for _ in range(n_estimators)
        ]

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64] | None = None,
        y_val: npt.NDArray[np.float64] | None = None,
        trace: bool | None = None,
        patience: int | None = None,
    ) -> ConvergenceHistory | None:
        """
        Train an ensemble of trees on the provided data.

        Args:
            X (npt.NDArray[np.float64]): Objects features matrix, array of shape (n_objects, n_features).
            y (npt.NDArray[np.float64]): Regression labels, array of shape (n_objects,).
            X_val (npt.NDArray[np.float64] | None, optional): Validation set of objects, array of shape (n_val_objects, n_features). Defaults to None.
            y_val (npt.NDArray[np.float64] | None, optional): Validation set of labels, array of shape (n_val_objects,). Defaults to None.
            trace (bool | None, optional): Whether to calculate rmsle while training. True by default if validation data is provided. Defaults to None.
            patience (int | None, optional): Number of training steps without decreasing the train loss (or validation if provided), after which to stop training. Defaults to None.

        Returns:
            ConvergenceHistory | None: Instance of `ConvergenceHistory` if `trace=True` or if validation data is provided.
        """
        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(self.n_estimators):
            # Fit each tree in the forest
            for tree in self.forest:
                tree.fit(X, y)

            # Predict on training and validation sets
            y_train_pred = self.predict(X)
            y_val_pred = self.predict(X_val) if X_val is not None else None

            # Calculate MSE
            train_loss = np.mean((y_train_pred - y) ** 2)
            val_loss = np.mean((y_val_pred - y_val) ** 2) if X_val is not None else None

            # Update losses history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Check for early stopping
            if patience is not None and epoch > patience:
                break

            # Check if validation loss improved
            if X_val is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            elif patience is not None and epoch > best_epoch + patience:
                break

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_estimators}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Optionally, return losses if tracing is enabled
            if trace is True or X_val is not None:
                return self.train_losses, self.val_losses

        return None

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Make prediction with ensemble of trees.

        All the trees make their own predictions which then are averaged.

        Args:
            X (npt.NDArray[np.float64]): Objects' features matrix, array of shape (n_objects, n_features).

        Returns:
            npt.NDArray[np.float64]: Predicted values, array of shape (n_objects,).
        """
        
        ...

    def dump(self, dirpath: str) -> None:
        """
        Save the trained model to the specified directory.

        Args:
            dirpath (str): Path to the directory where the model will be saved.
        """
        path = Path(dirpath)
        path.mkdir(parents=True)

        params = {"n_estimators": self.n_estimators}
        with (path / "params.json").open("w") as file:
            json.dump(params, file, indent=4)

        trees_path = path / "trees"
        trees_path.mkdir()
        for i, tree in enumerate(self.forest):
            joblib.dump(tree, trees_path / f"tree_{i:04d}.joblib")

    @classmethod
    def load(cls, dirpath: str) -> "RandomForestMSE":
        """
        Load a trained model from the specified directory.

        Args:
            dirpath (str): Path to the directory where the model is saved.

        Returns:
            RandomForestMSE: An instance of the loaded model.
        """
        with (Path(dirpath) / "params.json").open() as file:
            params = json.load(file)
        instance = cls(params["n_estimators"])

        trees_path = Path(dirpath) / "trees"

        instance.forest = [
            joblib.load(trees_path / f"tree_{i:04d}.joblib")
            for i in range(params["n_estimators"])
        ]

        return instance
