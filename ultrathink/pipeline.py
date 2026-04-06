"""UltraThink: cached feature + label pipeline for fast experiment iteration.

Usage:
    ut = UltraThink()

    # First call: computes (~30s). Subsequent: loads from cache (~1s).
    X, labels, kline, strat_info = ut.prepare("BTCUSDT", "2021-06-01", "2025-12-31")

    # With advanced features
    X, labels, kline, strat_info = ut.prepare(
        "BTCUSDT", "2021-06-01", "2025-12-31",
        extras=["fracdiff", "fft", "cross_asset"],
    )

    # Multi-symbol for global model
    X, labels, kline, strat_info = ut.prepare_multi(
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"], "2023-04-01", "2025-12-31",
    )
"""

import os
import time

import numpy as np
import pandas as pd

from ultrathink.cache import ParquetCache


class UltraThink:
    def __init__(self, data_dir: str = "data/merged", cache_dir: str = "data/cache"):
        self.data_dir = data_dir
        self.cache = ParquetCache(cache_dir)

    # ── Data Loading ────────────────────────────────────────────────────

    def load_klines(
        self, symbol: str, start: str, end: str,
        timeframes: list[str] = ("5m", "15m", "1h"),
    ) -> dict[str, pd.DataFrame]:
        kline = {}
        for tf in timeframes:
            path = os.path.join(self.data_dir, symbol, f"kline_{tf}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                kline[tf] = df.sort_index()[start:end]

        # Derive 4h from 1h if not available
        if "1h" in kline and "4h" not in kline:
            kline["4h"] = kline["1h"].resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
        return kline

    def load_extras(
        self, symbol: str, start: str, end: str,
    ) -> dict[str, pd.DataFrame]:
        extras = {}
        for name in ("tick_bar", "metrics", "funding_rate"):
            path = os.path.join(self.data_dir, symbol, f"{name}.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                extras[name] = df.sort_index()[start:end]
        return extras

    # ── Features ────────────────────────────────────────────────────────

    def features(
        self,
        symbol: str,
        start: str,
        end: str,
        target_tf: str = "15min",
        extras: list[str] | None = None,
        lag_top_n: int = 30,
        lag_depths: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute or load cached features.

        Args:
            extras: ["fracdiff", "fft", "cross_asset"] — additional feature sets
            lag_top_n: number of top-variance features to create lags for
            lag_depths: lag periods (default: [1..7] + change [1,2,4])
        """
        if extras is None:
            extras = []
        if lag_depths is None:
            lag_depths = list(range(1, 8))

        params = {
            "symbol": symbol, "start": start, "end": end,
            "target_tf": target_tf, "extras": sorted(extras),
            "lag_top_n": lag_top_n, "lag_depths": lag_depths,
            "version": "v2",
        }

        df, hit = self.cache.get(f"feat_{symbol}", params)
        if hit:
            return df

        t0 = time.time()
        kline = self.load_klines(symbol, start, end)
        extra_data = self.load_extras(symbol, start, end)

        # Base features
        from features.factory_v2 import generate_features_v2

        base = generate_features_v2(
            kline_data=kline,
            tick_bar=extra_data.get("tick_bar"),
            metrics=extra_data.get("metrics"),
            funding=extra_data.get("funding_rate"),
            target_tf=target_tf,
            progress=False,
        )

        parts = [base]

        # Advanced features
        if "fracdiff" in extras:
            parts.append(self._fracdiff_features(kline, target_tf))

        if "fft" in extras:
            parts.append(self._fft_features(kline, target_tf))

        if "cross_asset" in extras:
            ref_close = kline.get("15m", list(kline.values())[0])["close"]
            parts.append(self._cross_asset_features(ref_close, start, end))

        features = pd.concat(parts, axis=1)

        # Lag features from top-variance columns
        if lag_top_n > 0:
            top_cols = base.std().sort_values(ascending=False).head(lag_top_n).index.tolist()
            lag_dict = {}
            for lag in lag_depths:
                for col in top_cols:
                    lag_dict[f"{col}_lag{lag}"] = base[col].shift(lag)
            for col in top_cols[:10]:
                for lag in [1, 2, 4]:
                    lag_dict[f"{col}_chg{lag}"] = base[col] - base[col].shift(lag)
            features = pd.concat([features, pd.DataFrame(lag_dict, index=base.index)], axis=1)

        # Clean
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.loc[:, features.nunique() > 1]
        features = features.loc[:, features.notna().sum() > len(features) * 0.3]

        self.cache.put(f"feat_{symbol}", params, features)
        print(f"  [ultrathink] Features cached: {features.shape[1]} cols, {len(features)} rows ({time.time()-t0:.1f}s)")
        return features

    def _fracdiff_features(self, kline: dict, target_tf: str) -> pd.DataFrame:
        from validation.features.fracdiff import fracdiff

        df_15m = kline.get("15m", list(kline.values())[0])
        close = df_15m["close"]
        volume = df_15m["volume"]
        feats = {}
        for d_val in [0.3, 0.5]:
            fd = fracdiff(close, d_val)
            feats[f"fd_close_{d_val}"] = fd
            for lag in [1, 4, 12]:
                feats[f"fd_close_{d_val}_lag{lag}"] = fd.shift(lag)
        feats["fd_volume_05"] = fracdiff(volume, 0.5)
        feats["fd_range_04"] = fracdiff(df_15m["high"] - df_15m["low"], 0.4)
        result = pd.DataFrame(feats, index=close.index)
        if target_tf != "15min":
            result = result.resample(target_tf).last()
        return result

    def _fft_features(self, kline: dict, target_tf: str) -> pd.DataFrame:
        df_15m = kline.get("15m", list(kline.values())[0])
        close = df_15m["close"]
        values = close.values.astype(np.float64)
        n = len(values)
        feats = {}

        for w in [96, 384]:
            step = max(w // 8, 4)
            positions = np.arange(w, n, step)
            if len(positions) == 0:
                continue
            strides = (values.strides[0], values.strides[0])
            all_segs = np.lib.stride_tricks.as_strided(
                values, shape=(n - w + 1, w), strides=strides,
            )
            segs = all_segs[positions - w].copy()
            ramp = np.linspace(0, 1, w)[None, :]
            segs -= segs[:, 0:1] + ramp * (segs[:, -1:] - segs[:, 0:1])
            fft_vals = np.fft.rfft(segs, axis=1)
            mags = np.abs(fft_vals[:, 1:])
            dom_idx = np.argmax(mags, axis=1)
            quarter = max(mags.shape[1] // 4, 1)
            low_e = np.sum(mags[:, :quarter] ** 2, axis=1)
            high_e = np.sum(mags[:, -quarter:] ** 2, axis=1)
            total = low_e + high_e

            dp = np.full(n, np.nan)
            sr = np.full(n, np.nan)
            ph = np.full(n, np.nan)
            dp[positions] = w / (dom_idx + 1).astype(np.float64)
            sr[positions] = np.where(total > 0, low_e / total, 0.5)
            row_idx = np.arange(len(dom_idx))
            ph[positions] = np.angle(fft_vals[row_idx, dom_idx + 1])

            feats[f"fft_period_{w}"] = pd.Series(dp, index=close.index).ffill().values
            feats[f"fft_spec_ratio_{w}"] = pd.Series(sr, index=close.index).ffill().values
            feats[f"fft_phase_{w}"] = pd.Series(ph, index=close.index).ffill().values

        result = pd.DataFrame(feats, index=close.index)
        if target_tf != "15min":
            result = result.resample(target_tf).last()
        return result

    def _cross_asset_features(
        self, btc_close: pd.Series, start: str, end: str,
    ) -> pd.DataFrame:
        feats = {}
        for sym in ("ETHUSDT", "SOLUSDT"):
            path = os.path.join(self.data_dir, sym, "kline_15m.parquet")
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            alt = df["close"].sort_index()[start:end]
            alt = alt.reindex(btc_close.index, method="ffill")
            prefix = sym[:3].lower()

            ratio = alt / btc_close.replace(0, np.nan)
            feats[f"xa_{prefix}_ratio"] = ratio
            for w in [4, 12, 48]:
                feats[f"xa_{prefix}_ratio_chg_{w}"] = ratio.pct_change(w)
            btc_ret = btc_close.pct_change()
            alt_ret = alt.pct_change()
            for w in [24, 96]:
                feats[f"xa_{prefix}_corr_{w}"] = btc_ret.rolling(w).corr(alt_ret)
            for lag in [1, 2, 4]:
                feats[f"xa_{prefix}_lead_{lag}"] = alt_ret.shift(lag)

        return pd.DataFrame(feats, index=btc_close.index)

    # ── Labels ──────────────────────────────────────────────────────────

    def labels(
        self,
        symbol: str,
        start: str,
        end: str,
        target_tf: str = "15min",
        fee: float = 0.0008,
    ) -> pd.DataFrame:
        """Compute or load cached labels aligned to target_tf."""
        params = {
            "symbol": symbol, "start": start, "end": end,
            "target_tf": target_tf, "fee": fee,
            "version": "multi_tbm_v2",
        }

        df, hit = self.cache.get(f"label_{symbol}", params)
        if hit:
            return df

        t0 = time.time()
        kline = self.load_klines(symbol, start, end)

        from labeling.multi_tbm_v2 import generate_multi_tbm_v2

        lr = generate_multi_tbm_v2(kline, fee_pct=fee, progress=False)

        # Merge all strategies into single DataFrame aligned to target_tf
        base_strat = "intraday" if "intraday" in lr else next(iter(lr))
        frames = [lr[base_strat]]
        for name, sdf in lr.items():
            if name == base_strat:
                continue
            resampled = sdf.resample(target_tf).ffill()
            new_cols = [c for c in resampled.columns if c not in lr[base_strat].columns]
            if new_cols:
                frames.append(resampled[new_cols])
        labels = pd.concat(frames, axis=1)

        self.cache.put(f"label_{symbol}", params, labels)
        print(f"  [ultrathink] Labels cached: {labels.shape[1]} cols ({time.time()-t0:.1f}s)")
        return labels

    # ── Prepare (one-call) ──────────────────────────────────────────────

    def prepare(
        self,
        symbol: str,
        start: str,
        end: str,
        target_tf: str = "15min",
        extras: list[str] | None = None,
        fee: float = 0.0008,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
        """One-call preparation for experiments.

        Returns:
            X: features DataFrame
            labels: labels DataFrame (tbm_, mae_, mfe_, rar_, wgt_ columns)
            kline: dict of kline DataFrames
            strat_info: list of strategy metadata dicts
        """
        t0 = time.time()

        X = self.features(symbol, start, end, target_tf=target_tf, extras=extras)
        labels = self.labels(symbol, start, end, target_tf=target_tf, fee=fee)
        kline = self.load_klines(symbol, start, end)

        # Align
        common = X.index.intersection(labels.index)
        X = X.loc[common]
        labels = labels.loc[common]

        # Strategy info
        tc_cols = sorted([c for c in labels.columns if c.startswith("tbm_")])
        strat_info = []
        for c in tc_cols:
            parts = c.replace("tbm_", "").split("_")
            strat_info.append({"style": parts[0], "dir": parts[1]})

        elapsed = time.time() - t0
        print(f"  [ultrathink] Ready: {X.shape[1]} features × {len(X)} rows, "
              f"{len(tc_cols)} strategies ({elapsed:.1f}s)")
        return X, labels, kline, strat_info

    # ── Multi-Symbol ────────────────────────────────────────────────────

    def prepare_multi(
        self,
        symbols: list[str],
        start: str,
        end: str,
        target_tf: str = "15min",
        extras: list[str] | None = None,
        fee: float = 0.0008,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
        """Prepare data for multiple symbols (global model).

        Returns concatenated X and labels with a 'symbol' column for identification.
        """
        all_X, all_labels = [], []

        for sym in symbols:
            try:
                X, labels, _, _ = self.prepare(sym, start, end, target_tf, extras, fee)
                X["_symbol"] = sym
                labels["_symbol"] = sym
                all_X.append(X)
                all_labels.append(labels)
            except Exception as e:
                print(f"  [ultrathink] {sym}: skipped ({e})")
                continue

        X_all = pd.concat(all_X).sort_index()
        labels_all = pd.concat(all_labels).sort_index()

        tc_cols = sorted([c for c in labels_all.columns if c.startswith("tbm_")])
        strat_info = []
        for c in tc_cols:
            parts = c.replace("tbm_", "").split("_")
            strat_info.append({"style": parts[0], "dir": parts[1]})

        print(f"\n  [ultrathink] Global: {len(symbols)} symbols, "
              f"{X_all.shape[1]-1} features × {len(X_all)} rows")
        return X_all, labels_all, strat_info

    # ── Cache Management ────────────────────────────────────────────────

    def cache_info(self) -> list[dict]:
        return self.cache.list()

    def cache_clear(self, what: str | None = None) -> int:
        """Clear cache. what='features', 'labels', or None for all."""
        ns = f"feat_" if what == "features" else f"label_" if what == "labels" else None
        return self.cache.clear(ns)
