"""
ultraTM Portfolio Strategy — 5-Coin NautilusTrader Execution

Loads saved production models for BTC/BCH/ONT/LINK/ATOM.
Generates signals on each 15m bar, allocates via Position Manager.

For paper/live trading via NautilusTrader or direct API.
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from ple.model_v4 import PLEv4
from ple.model_v3 import partition_features


class PortfolioSignalGenerator:
    """Generate trading signals for 5-coin portfolio.

    Usage:
        gen = PortfolioSignalGenerator("models/production_v2")
        signals = gen.predict(features_dict)
        # signals = [{"asset": "BTCUSDT", "direction": 1, "probability": 0.72, ...}, ...]
    """

    def __init__(self, model_dir: str = "models/production_v2", device: str = "cuda"):
        self.device = device
        self.models = {}
        self.configs = {}

        config_path = Path(model_dir) / "config.json"
        with open(config_path) as f:
            self.global_config = json.load(f)

        for coin in self.global_config["coins"]:
            path = Path(model_dir) / f"{coin.lower()}.pt"
            if not path.exists():
                continue

            checkpoint = torch.load(path, map_location=device, weights_only=False)
            pt = checkpoint["partitions"]
            ns = checkpoint["n_strategies"]
            si = checkpoint["strat_info"]
            feat_cols = checkpoint["feature_cols"]

            model = PLEv4(
                feature_partitions=pt,
                n_account_features=4,
                n_strategies=ns,
                expert_hidden=256,
                expert_output=128,
                fusion_dim=256,
                dropout=0.2,
                use_vsn=False,
            )
            model.load_state_dict(checkpoint["model_state"])
            model = model.to(device).eval()

            self.models[coin] = model
            self.configs[coin] = {
                "partitions": pt,
                "strat_info": si,
                "feature_cols": feat_cols,
                "n_strategies": ns,
            }

        print(f"Loaded {len(self.models)} models: {list(self.models.keys())}")

    def predict(
        self,
        features: dict[str, np.ndarray],
        sma_above: dict[str, bool],
        fee: float = 0.0008,
    ) -> list[dict]:
        """Generate signals for all coins.

        Args:
            features: {coin: feature_array (1, n_features)} per coin
            sma_above: {coin: True/False} price above SMA50
            fee: trading fee

        Returns:
            List of signal dicts sorted by EV descending.
        """
        holds_map = {"scalp": 1, "intraday": 4, "daytrade": 48, "swing": 168}
        all_signals = []

        for coin, model in self.models.items():
            if coin not in features:
                continue

            x = torch.tensor(features[coin], dtype=torch.float32).to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)

            acc = torch.zeros(1, 4, device=self.device)

            with torch.no_grad():
                out = model(x, acc)
                probs = out["label_probs"].cpu().numpy()[0]
                mfe_pred = out["mfe_pred"].cpu().numpy()[0]
                mae_pred = out["mae_pred"].cpu().numpy()[0]

            si = self.configs[coin]["strat_info"]
            above = sma_above.get(coin, True)
            lt = 0.40 if above else 0.55
            st = 0.55 if above else 0.40

            for j, info in enumerate(si):
                is_long = info["dir"] == "long"
                thresh = lt if is_long else st

                if probs[j] < thresh:
                    continue

                p = probs[j]
                rew = max(abs(mfe_pred[j]), 0.001)
                rsk = max(abs(mae_pred[j]), 0.001)
                ev = p * rew - (1 - p) * rsk - fee

                if ev <= 0:
                    continue

                all_signals.append({
                    "asset": coin,
                    "direction": 1 if is_long else -1,
                    "probability": float(p),
                    "mfe": float(rew),
                    "mae": float(rsk),
                    "ev": float(ev),
                    "strategy": f"{info['style']}_{info['dir']}",
                    "hold_bars": holds_map.get(info["style"], 12),
                })

        # Sort by EV descending
        all_signals.sort(key=lambda x: x["ev"], reverse=True)
        return all_signals

    @property
    def coins(self) -> list[str]:
        return list(self.models.keys())
