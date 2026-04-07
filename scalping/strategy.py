"""Production Scalping Strategy — Universal GBM Multi-Coin.

Architecture:
  - 1 Universal GBM model (dual long/short)
  - 12+ altcoins simultaneously
  - 15min hold (3 x 5m bars), close-to-close exit
  - Taker fee compatible (+0.094%/trade avg)

Usage:
  strategy = ScalpingStrategy()
  strategy.train(coins_data, start, end)
  signals = strategy.predict(current_features)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass


# Validated profitable coins (walk-forward OOS, Taker positive)
SCALP_COINS = [
    'ENAUSDT', 'WLDUSDT', 'JUPUSDT', 'ARBUSDT', 'ONDOUSDT',
    '1000PEPEUSDT', 'ORDIUSDT', 'TIAUSDT', 'EDUUSDT',
    'CELOUSDT', 'DYDXUSDT', 'ALICEUSDT',
]

FEATURE_NAMES = [
    'coin_id', 'ret_1', 'ret_3', 'body', 'range_pct',
    'vol_5', 'vol_20', 'vol_accel', 'vol_ratio',
    'tc_ratio', 'buy_ratio', 'range_pos', 'vwap_dist',
    'hour', 'mr_strength',
]

GBM_PARAMS = {
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


@dataclass
class ScalpSignal:
    coin: str
    direction: int  # +1 long, -1 short
    prob: float
    timestamp: pd.Timestamp


def build_features(k5: pd.DataFrame, tick_5m: pd.DataFrame, coin_id: int) -> pd.DataFrame:
    """Build 15 features from 5m OHLCV + tick data."""
    cc = k5['close']; vv = k5['volume']
    ret = cc.pct_change()
    ret3 = (cc / cc.shift(3) - 1).values

    f = pd.DataFrame(index=k5.index)
    f['coin_id'] = coin_id
    f['ret_1'] = ret
    f['ret_3'] = ret3
    f['body'] = (k5['close'] - k5['open']) / (k5['high'] - k5['low'] + 1e-10)
    f['range_pct'] = (k5['high'] - k5['low']) / k5['close']
    f['vol_5'] = ret.rolling(5).std()
    f['vol_20'] = ret.rolling(20).std()
    f['vol_accel'] = f['vol_5'] / (f['vol_20'] + 1e-10)
    f['vol_ratio'] = vv / (vv.rolling(12).mean() + 1e-10)
    f['tc_ratio'] = (tick_5m['trade_count'] / (tick_5m['trade_count'].rolling(12).mean() + 1e-10)).reindex(k5.index)
    f['buy_ratio'] = (tick_5m['buy_volume'] / (tick_5m['buy_volume'] + tick_5m['sell_volume'] + 1e-10)).reindex(k5.index)
    f['range_pos'] = (cc - k5['low'].rolling(20).min()) / (k5['high'].rolling(20).max() - k5['low'].rolling(20).min() + 1e-10)
    f['vwap_dist'] = cc / ((cc * vv).rolling(20).sum() / (vv.rolling(20).sum() + 1e-10)) - 1
    f['hour'] = k5.index.hour
    f['mr_strength'] = np.abs(ret3)

    return f.replace([np.inf, -np.inf], np.nan)


class ScalpingStrategy:
    """Universal Multi-Coin Scalping Strategy.

    Trains dual GBM (long/short) on multiple coins simultaneously.
    Predicts direction with confidence threshold.
    """

    def __init__(self, prob_threshold: float = 0.58, n_ensemble: int = 2):
        self.prob_threshold = prob_threshold
        self.n_ensemble = n_ensemble
        self.models_long = []
        self.models_short = []
        self.coin_map = {}

    def train(self, train_data: dict[str, pd.DataFrame], val_data: dict[str, pd.DataFrame]):
        """Train on multiple coins.

        Args:
            train_data: {coin: DataFrame with features + 'label' column}
            val_data: same format
        """
        # Combine all coins
        all_tr = pd.concat(train_data.values())
        all_va = pd.concat(val_data.values())

        X_tr = np.nan_to_num(all_tr[FEATURE_NAMES].values, 0)
        y_tr = all_tr['label'].values
        X_va = np.nan_to_num(all_va[FEATURE_NAMES].values, 0)
        y_va = all_va['label'].values

        self.models_long = []
        self.models_short = []

        for seed in range(self.n_ensemble):
            params = {**GBM_PARAMS, 'seed': seed * 42}

            # Long model
            gl = lgb.train(
                params,
                lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES),
                num_boost_round=500,
                valid_sets=[lgb.Dataset(X_va, y_va, feature_name=FEATURE_NAMES)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            )
            self.models_long.append(gl)

            # Short model
            gs = lgb.train(
                {**params, 'seed': seed * 42 + 1000},
                lgb.Dataset(X_tr, 1 - y_tr, feature_name=FEATURE_NAMES),
                num_boost_round=500,
                valid_sets=[lgb.Dataset(X_va, 1 - y_va, feature_name=FEATURE_NAMES)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
            )
            self.models_short.append(gs)

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """Predict long/short probabilities.

        Returns: (prob_long, prob_short)
        """
        x = np.nan_to_num(features.reshape(1, -1), 0)
        pl = np.mean([m.predict(x)[0] for m in self.models_long])
        ps = np.mean([m.predict(x)[0] for m in self.models_short])
        return pl, ps

    def get_signal(self, features: np.ndarray, coin: str, timestamp: pd.Timestamp) -> ScalpSignal | None:
        """Get trading signal if confidence exceeds threshold."""
        pl, ps = self.predict(features)

        if pl >= self.prob_threshold and pl > ps:
            return ScalpSignal(coin=coin, direction=1, prob=pl, timestamp=timestamp)
        elif ps >= self.prob_threshold and ps > pl:
            return ScalpSignal(coin=coin, direction=-1, prob=ps, timestamp=timestamp)
        return None
