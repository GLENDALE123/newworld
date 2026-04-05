import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


class TradingModel:
    def __init__(
        self,
        iterations: int = 500,
        depth: int = 6,
        learning_rate: float = 0.05,
        task_type: str = "CPU",
    ):
        self.params = {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "task_type": task_type,
            "loss_function": "Logloss",
            "verbose": 0,
            "random_seed": 42,
        }
        self.model: CatBoostClassifier | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model = CatBoostClassifier(**self.params)
        y_binary = (y == 1.0).astype(int)
        self.model.fit(X, y_binary)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        preds_binary = self.model.predict(X).flatten()
        return np.where(preds_binary == 1, 1.0, -1.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

    def walk_forward(
        self,
        data: pd.DataFrame,
        train_window: int,
        val_window: int,
    ) -> dict:
        feature_cols = [c for c in data.columns if c != "label"]
        all_preds = []
        all_actuals = []
        all_indices = []

        start = 0
        while start + train_window + val_window <= len(data):
            train_end = start + train_window
            val_end = train_end + val_window

            train_slice = data.iloc[start:train_end]
            val_slice = data.iloc[train_end:val_end]

            self.train(train_slice[feature_cols], train_slice["label"])
            preds = self.predict(val_slice[feature_cols])

            all_preds.extend(preds)
            all_actuals.extend(val_slice["label"].values)
            all_indices.extend(val_slice.index.tolist())

            start += val_window

        return {
            "predictions": np.array(all_preds),
            "actuals": np.array(all_actuals),
            "indices": all_indices,
        }
