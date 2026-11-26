"""Features, three forecasting methods and walk-forward testing.

Honest setup: every feature uses only information that exists at forecast time.
Lags go back 7, 14, 21 and 28 days, so a forecast for any day of next week uses
only actuals up to the end of this week. Holidays and campaigns are on the
calendar in advance, so they are allowed. Incidents are not, and no model sees them.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

SLOTS = 32
FEATURES = ["slot", "dow", "month", "t", "is_holiday", "is_after_holiday", "is_campaign",
            "days_to_holiday", "lag7_ratio", "lag14_ratio", "lag21_ratio", "lag28_ratio",
            "special_in_lags", "base"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "slot"]).reset_index(drop=True).copy()
    df["special"] = ((df.is_holiday + df.is_after_holiday + df.is_campaign) > 0).astype(int)
    for k in (1, 2, 3, 4):
        df[f"lag{7 * k}"] = df["offered"].shift(SLOTS * 7 * k)
        df[f"special_lag{7 * k}"] = df["special"].shift(SLOTS * 7 * k)
    lags = ["lag7", "lag14", "lag21", "lag28"]
    df["seasonal_avg"] = df[lags].mean(axis=1)
    # a cleaner baseline: average only of lags that were ordinary days
    ok = df[[f"special_lag{k}" for k in (7, 14, 21, 28)]].eq(0).to_numpy()
    vals = df[lags].to_numpy()
    clean = np.where(ok, vals, 0.0)
    n_ok = ok.sum(axis=1)
    # mean of ordinary lags; if every lag was a special day, fall back to all lags
    df["base"] = np.where(n_ok > 0, clean.sum(axis=1) / np.maximum(n_ok, 1), vals.mean(axis=1))
    for k in (7, 14, 21, 28):
        df[f"lag{k}_ratio"] = df[f"lag{k}"] / (df["base"] + 1)
    df["special_in_lags"] = df[[f"special_lag{k}" for k in (7, 14, 21, 28)]].sum(axis=1)
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["t"] = (df["date"] - df["date"].min()).dt.days
    hol_days = df.loc[df.is_holiday == 1, "date"].drop_duplicates().to_numpy()
    d = df["date"].to_numpy()
    dist = np.abs(d[:, None].astype("datetime64[D]") - hol_days[None, :].astype("datetime64[D]"))
    df["days_to_holiday"] = np.minimum(dist.min(axis=1).astype(int), 30)
    return df.dropna(subset=["lag28"]).reset_index(drop=True)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Return test with three forecast columns: naive, seasonal, ml."""
    out = test.copy()
    out["naive"] = out["lag7"]
    out["seasonal"] = out["seasonal_avg"]
    model = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=2)
    y = train["offered"] / (train["base"] + 1)
    model.fit(train[FEATURES], y)
    out["ml"] = np.maximum(model.predict(test[FEATURES]) * (test["base"] + 1) - 1, 0)
    return out


def walk_forward(df: pd.DataFrame, folds: int = 8, block_days: int = 28) -> pd.DataFrame:
    days = np.sort(df["date"].unique())
    first_test = len(days) - folds * block_days
    parts = []
    for f in range(folds):
        lo = days[first_test + f * block_days]
        hi = days[first_test + (f + 1) * block_days - 1]
        train = df[df["date"] < lo]
        test = df[(df["date"] >= lo) & (df["date"] <= hi)]
        res = fit_predict(train, test)
        res["fold"] = f + 1
        parts.append(res)
    return pd.concat(parts, ignore_index=True)


def wape(actual, forecast) -> float:
    """Weighted absolute percentage error: total absolute miss / total actual."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float(np.abs(actual - forecast).sum() / actual.sum())


def bias(actual, forecast) -> float:
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    return float((forecast.sum() - actual.sum()) / actual.sum())
