from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Market
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"

    # TBM parameters
    tbm_pt_multiplier: float = 1.0
    tbm_sl_multiplier: float = 1.0
    tbm_max_holding_bars: int = 4
    volatility_span: int = 24

    # Risk
    position_size_pct: float = 0.02
    initial_capital: float = 100_000.0

    # CatBoost
    train_window_months: int = 3
    val_window_months: int = 1
    catboost_iterations: int = 500
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05

    # Data path - existing data
    data_dir: str = "data/merged"

    model_config = {"env_prefix": "ULTRATM_"}
