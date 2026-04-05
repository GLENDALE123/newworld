import pandas as pd
from config.settings import Settings
from features.technical import compute_technical_features
from labeling.tbm import TripleBarrierLabeler


class FeaturePipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.labeler = TripleBarrierLabeler(
            pt_multiplier=settings.tbm_pt_multiplier,
            sl_multiplier=settings.tbm_sl_multiplier,
            max_holding_bars=settings.tbm_max_holding_bars,
        )

    def build(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        features = compute_technical_features(ohlcv)
        labels = self.labeler.label(ohlcv)
        features["label"] = labels
        features = features.dropna()
        return features
