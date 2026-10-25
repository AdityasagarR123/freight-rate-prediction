"""Cleaning and feature engineering shared by training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]
EPOCH = pd.Timestamp("2025-01-01")

# Label outliers sit at ~0.2-0.5x or ~2-6x the expected rate while clean rows
# have a log-residual std of ~0.04, so |log residual| > 0.25 is a >6-sigma event.
OUTLIER_LOG_THRESHOLD = 0.25
QUARTER_RAMP_DAYS = 21


def load(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def daily_market(frame: pd.DataFrame) -> pd.DataFrame:
    """Date-level market signals: market_index is (almost) a per-day series.

    Within-day std of market_index is ~0.025 vs ~0.17 between days, so the daily
    mean is a reliable imputation value and a denoised market feature.
    """
    daily = frame.groupby("date").agg(
        day_market_index=("market_index", "mean"),
        day_quote_signal=("quote_signal", "mean"),
    )
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full).interpolate(limit_direction="both")
    daily.index.name = "date"
    return daily


def clean(frame: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fix the data-quality issues found in EDA. Never drops rows."""
    out = frame.copy()
    if daily is None:
        daily = daily_market(out)
    out = out.join(daily, on="date")

    # Negative weights are sign errors: |w| matches the positive weight distribution.
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["weight"] = out["weight"].abs()
    out["weight"] = out["weight"].fillna(out.groupby("equipment")["weight"].transform("median"))
    out["weight"] = out["weight"].fillna(31_500.0)

    out["market_index_missing"] = out["market_index"].isna().astype(int)
    out["market_index"] = out["market_index"].fillna(out["day_market_index"])
    return out


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["log_distance"] = np.log(out["distance"])
    out["log_weight"] = np.log(out["weight"])
    out["equipment_code"] = out["equipment"].map({name: i for i, name in enumerate(EQUIPMENT)}).astype(int)
    out["log_market_index"] = np.log(out["market_index"])
    out["log_day_market_index"] = np.log(out["day_market_index"])
    out["log_quote_signal"] = np.log(out["quote_signal"])

    date = out["date"]
    quarter_start = date.dt.to_period("Q").dt.start_time
    quarter_end = date.dt.to_period("Q").dt.end_time.dt.normalize()
    out["day_of_week"] = date.dt.day_of_week
    out["day_of_quarter"] = (date - quarter_start).dt.days
    out["days_to_quarter_end"] = (quarter_end - date).dt.days
    out["days_to_month_end"] = (date + pd.offsets.MonthEnd(0) - date).dt.days
    out["day_of_month"] = date.dt.day
    out["days_since_start"] = (date - EPOCH).dt.days
    return out


def prepare(frame: pd.DataFrame, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    return add_features(clean(frame, daily))


# Load-level features only. Date-level series (daily market mean, calendar
# position) let trees memorise individual training days, which does not
# transfer to unseen dates; the time component is modelled parametrically instead.
STATIC_FEATURES = [
    "log_distance",
    "log_weight",
    "equipment_code",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "log_market_index",
    "log_quote_signal",
]

GBM_FEATURES = [
    "log_distance",
    "log_weight",
    "equipment_code",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "log_market_index",
    "log_day_market_index",
    "log_quote_signal",
    "day_of_week",
    "day_of_quarter",
    "days_to_quarter_end",
    "days_to_month_end",
]


def linear_design(frame: pd.DataFrame, with_trend: bool = True) -> pd.DataFrame:
    """Design matrix for the robust log-linear model (baseline and outlier filter)."""
    design = pd.DataFrame(index=frame.index)
    design["log_distance"] = frame["log_distance"]
    design["log_distance_sq"] = frame["log_distance"] ** 2
    design["weight_10k"] = frame["weight"] / 1e4
    design["reefer"] = (frame["equipment"] == "Reefer").astype(float)
    design["flatbed"] = (frame["equipment"] == "Flatbed").astype(float)
    design["log_market_index"] = frame["log_market_index"]
    design["log_quote_signal"] = frame["log_quote_signal"]
    # Quarter-end ramp: rates rise roughly linearly over the last ~3 weeks of each
    # quarter (+~4% by the final days) and reset on day 1 of the next quarter.
    design["quarter_end_ramp"] = np.clip(QUARTER_RAMP_DAYS - frame["days_to_quarter_end"], 0, QUARTER_RAMP_DAYS) / QUARTER_RAMP_DAYS
    if with_trend:
        design["trend_years"] = frame["days_since_start"] / 365.0
    return design
