"""Production Intraday (단타) Strategy — Regression GBM + Checkpoint Exit.

Architecture:
  - 3 Regression GBM models (1h, 2h, 4h targets)
  - 2h model for entry signal
  - 1h model for checkpoint exit decision
  - Dynamic holding: 1h/2h/4h based on model agreement
  - 20+ altcoins, BTC cross features

Entry: |pred_2h| > threshold
Exit (Checkpoint):
  - 1h+2h agree + |pred|>0.004 → hold 4h
  - 1h+2h agree → hold 2h
  - 1h+2h disagree → exit at 1h

3-Tier system:
  Volume:  Reg + checkpoint               → +0.329%/trade, 30K trades
  Quality: Reg+Cls agree + checkpoint     → +1.060%/trade, 4.7K trades
  Sniper:  Ultra (reg>0.003 + cls>0.57)   → +4.919%/trade, 951 trades
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# Tier A: high return, higher risk (~0.4%+ per trade)
TIER_A_COINS = ['SEIUSDT', 'TIAUSDT', 'DOTUSDT', '1000BONKUSDT', 'FETUSDT', 'ENAUSDT']

# Tier B: balanced (~0.2-0.4% per trade)
TIER_B_COINS = ['TAOUSDT', 'APTUSDT', 'BCHUSDT', 'ARBUSDT', 'INJUSDT', 'ORDIUSDT']

# Tier C: stable, lower return (~0.1-0.2% per trade)
TIER_C_COINS = ['TRUUSDT', 'LTCUSDT', '1INCHUSDT', 'CHRUSDT', 'GALAUSDT', 'SNXUSDT', 'JTOUSDT', 'AAVEUSDT']

# All validated coins (20/20 positive in rolling WF)
INTRADAY_COINS = TIER_A_COINS + TIER_B_COINS + TIER_C_COINS

FEATURE_NAMES = [
    'coin_id', 'oi_chg_1h', 'oi_chg_4h', 'oi_level_z',
    'funding', 'taker_ls_z',
    'ret_1h', 'ret_4h', 'ret_24h', 'range_pos_288', 'vol_60',
    'buy_ratio', 'btc_ret_1h', 'btc_oi_chg', 'btc_vol',
    'alt_btc_z', 'hour',
]

GBM_PARAMS = {
    'objective': 'regression',
    'metric': 'l1',
    'learning_rate': 0.01,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 200,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 2.0,
    'reg_lambda': 5.0,
    'verbose': -1,
}

HORIZONS = {'1h': 12, '2h': 24, '4h': 48}  # 5m bars


@dataclass
class IntradaySignal:
    coin: str
    direction: int  # +1 long, -1 short
    predicted_return: float
    hold_hours: str  # '1h', '2h', '4h'
    hold_bars: int
    timestamp: pd.Timestamp


CLS_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 200,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 2.0,
    'reg_lambda': 5.0,
    'verbose': -1,
}


class IntradayStrategy:
    """Multi-horizon regression GBM with checkpoint exit + optional classification ensemble.

    Tiers:
      Volume:  entry_threshold=0.002, no cls filter
      Quality: entry_threshold=0.002, cls_threshold=0.55
      Sniper:  entry_threshold=0.003, cls_threshold=0.57
    """

    def __init__(self, entry_threshold: float = 0.002, strong_threshold: float = 0.004,
                 cls_threshold: float = 0.0, n_ensemble: int = 2, use_catboost: bool = True):
        self.entry_threshold = entry_threshold
        self.strong_threshold = strong_threshold
        self.cls_threshold = cls_threshold  # 0 = no cls filter
        self.n_ensemble = n_ensemble
        self.use_catboost = use_catboost and HAS_CATBOOST
        self.models = {}
        self.catboost_models = {}  # CatBoost regression per horizon
        self.cls_models_long = []
        self.cls_models_short = []

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Train 3 horizon models on combined coin data."""
        for horizon, bars in HORIZONS.items():
            target_col = f'target_{horizon}'
            tr = train_data.dropna(subset=[target_col])
            va = val_data.dropna(subset=[target_col])

            X_tr = np.nan_to_num(tr[FEATURE_NAMES].values, 0)
            y_tr = tr[target_col].values
            X_va = np.nan_to_num(va[FEATURE_NAMES].values, 0)
            y_va = va[target_col].values

            models = []
            for seed in range(self.n_ensemble):
                params = {**GBM_PARAMS, 'seed': seed * 42}
                gbm = lgb.train(
                    params,
                    lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES),
                    num_boost_round=500,
                    valid_sets=[lgb.Dataset(X_va, y_va, feature_name=FEATURE_NAMES)],
                    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
                )
                models.append(gbm)
            self.models[horizon] = models

            # CatBoost for same horizon (+79% improvement in ensemble)
            if self.use_catboost:
                cb = CatBoostRegressor(
                    iterations=300, learning_rate=0.01, depth=6,
                    l2_leaf_reg=5, random_seed=42, verbose=0,
                    early_stopping_rounds=15,
                )
                cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
                self.catboost_models[horizon] = cb

        # Train classification models (for Reg+Cls ensemble)
        if self.cls_threshold > 0:
            target_col = 'cls_target_2h'
            if target_col in train_data.columns:
                tr_cls = train_data.dropna(subset=[target_col])
                va_cls = val_data.dropna(subset=[target_col])
                X_tr = np.nan_to_num(tr_cls[FEATURE_NAMES].values, 0)
                X_va = np.nan_to_num(va_cls[FEATURE_NAMES].values, 0)
                y_tr = tr_cls[target_col].values
                y_va = va_cls[target_col].values

                for seed in range(self.n_ensemble):
                    params_l = {**CLS_PARAMS, 'seed': seed * 42}
                    gl = lgb.train(params_l, lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES),
                        num_boost_round=500,
                        valid_sets=[lgb.Dataset(X_va, y_va, feature_name=FEATURE_NAMES)],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
                    self.cls_models_long.append(gl)

                    params_s = {**CLS_PARAMS, 'seed': seed * 42 + 1000}
                    gs = lgb.train(params_s, lgb.Dataset(X_tr, 1 - y_tr, feature_name=FEATURE_NAMES),
                        num_boost_round=500,
                        valid_sets=[lgb.Dataset(X_va, 1 - y_va, feature_name=FEATURE_NAMES)],
                        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
                    self.cls_models_short.append(gs)

    def predict(self, features: np.ndarray) -> dict[str, float]:
        """Predict return at all horizons + classification probabilities.

        If CatBoost available, returns LGB+CB ensemble average.
        """
        x = np.nan_to_num(features.reshape(1, -1), 0)
        result = {}
        for h, ms in self.models.items():
            lgb_pred = np.mean([m.predict(x)[0] for m in ms])
            if h in self.catboost_models:
                cb_pred = self.catboost_models[h].predict(x)[0]
                result[h] = (lgb_pred + cb_pred) / 2  # ensemble
            else:
                result[h] = lgb_pred
        if self.cls_models_long:
            result['cls_long'] = np.mean([m.predict(x)[0] for m in self.cls_models_long])
            result['cls_short'] = np.mean([m.predict(x)[0] for m in self.cls_models_short])
        return result

    def get_signal(self, features: np.ndarray, coin: str,
                   timestamp: pd.Timestamp) -> IntradaySignal | None:
        """Get trading signal with checkpoint-based hold time."""
        preds = self.predict(features)
        pred_2h = preds['2h']
        pred_1h = preds['1h']

        if abs(pred_2h) < self.entry_threshold:
            return None

        direction = 1 if pred_2h > 0 else -1

        # Classification filter (for Quality/Sniper tiers)
        if self.cls_threshold > 0 and 'cls_long' in preds:
            cls_conf = preds['cls_long'] if direction == 1 else preds['cls_short']
            if cls_conf < self.cls_threshold:
                return None

        # Checkpoint logic
        agree = (pred_1h > 0) == (pred_2h > 0)
        if self.cls_threshold > 0 and 'cls_long' in preds:
            cls_conf = preds['cls_long'] if direction == 1 else preds['cls_short']
            if agree and abs(pred_2h) > self.strong_threshold and cls_conf > 0.57:
                hold = '4h'
            elif agree and cls_conf > 0.53:
                hold = '2h'
            else:
                hold = '1h'
        else:
            if agree and abs(pred_2h) > self.strong_threshold:
                hold = '4h'
            elif agree:
                hold = '2h'
            else:
                hold = '1h'

        return IntradaySignal(
            coin=coin,
            direction=direction,
            predicted_return=pred_2h,
            hold_hours=hold,
            hold_bars=HORIZONS[hold],
            timestamp=timestamp,
        )


# Convenience constructors for each tier
def create_volume_strategy() -> IntradayStrategy:
    """Volume tier: Reg + checkpoint, +0.329%/trade."""
    return IntradayStrategy(entry_threshold=0.002, strong_threshold=0.004, cls_threshold=0.0)


def create_quality_strategy() -> IntradayStrategy:
    """Quality tier: Reg+Cls agree + checkpoint, +1.060%/trade."""
    return IntradayStrategy(entry_threshold=0.002, strong_threshold=0.004, cls_threshold=0.55)


def create_sniper_strategy() -> IntradayStrategy:
    """Sniper tier: Ultra selective, +4.919%/trade."""
    return IntradayStrategy(entry_threshold=0.003, strong_threshold=0.004, cls_threshold=0.57)


def tiered_position_size(predicted_return: float, base_size: float = 500) -> float:
    """Tiered position sizing: larger trades for stronger predictions.

    +12% improvement over fixed sizing in backtests.
    """
    ap = abs(predicted_return)
    if ap < 0.003:
        return base_size * 0.6  # small
    elif ap < 0.006:
        return base_size        # medium
    else:
        return base_size * 1.6  # large
