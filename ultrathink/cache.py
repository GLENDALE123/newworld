"""Deterministic parquet cache with parameter hashing."""

import hashlib
import json
import os
import time

import pandas as pd


class ParquetCache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, namespace: str, params: dict) -> str:
        s = json.dumps(params, sort_keys=True, default=str)
        h = hashlib.sha256(s.encode()).hexdigest()[:12]
        return f"{namespace}_{h}"

    def get(self, namespace: str, params: dict) -> tuple[pd.DataFrame | None, bool]:
        key = self._key(namespace, params)
        path = os.path.join(self.cache_dir, f"{key}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            return df, True
        return None, False

    def put(self, namespace: str, params: dict, df: pd.DataFrame) -> str:
        key = self._key(namespace, params)
        path = os.path.join(self.cache_dir, f"{key}.parquet")
        meta_path = os.path.join(self.cache_dir, f"{key}.json")
        df.to_parquet(path, index=True)
        meta = {
            "namespace": namespace,
            "params": params,
            "rows": len(df),
            "cols": df.shape[1],
            "columns_sample": list(df.columns)[:30],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return path

    def exists(self, namespace: str, params: dict) -> bool:
        key = self._key(namespace, params)
        return os.path.exists(os.path.join(self.cache_dir, f"{key}.parquet"))

    def clear(self, namespace: str | None = None) -> int:
        removed = 0
        for f in os.listdir(self.cache_dir):
            if namespace and not f.startswith(namespace):
                continue
            os.remove(os.path.join(self.cache_dir, f))
            removed += 1
        return removed

    def list(self) -> list[dict]:
        entries = []
        for f in sorted(os.listdir(self.cache_dir)):
            if f.endswith(".json"):
                with open(os.path.join(self.cache_dir, f)) as fh:
                    entries.append(json.load(fh))
        return entries
