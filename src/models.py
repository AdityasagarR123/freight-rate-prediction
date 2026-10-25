"""Candidate models. All of them fit log(posted_rate) and predict exp(.) (a median-style estimate)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from features import GBM_FEATURES, OUTLIER_LOG_THRESHOLD, STATIC_FEATURES, linear_design

GBM_PARAMS = dict(
    objective="huber",
    alpha=0.1,
    learning_rate=0.03,
    n_estimators=2500,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
    random_state=7,
)


def flag_label_outliers(train: pd.DataFrame, score: pd.DataFrame | None = None) -> np.ndarray:
    """Boolean mask of rows in `score` (default: `train`) whose label is inconsistent
    with a robust log-linear fit on `train`."""
    score = train if score is None else score
    huber = HuberRegressor(max_iter=2000, epsilon=1.35).fit(linear_design(train), np.log(train["posted_rate"]))
    residual = np.log(score["posted_rate"]) - huber.predict(linear_design(score))
    return np.abs(residual.to_numpy()) > OUTLIER_LOG_THRESHOLD


class LinearModel:
    """Robust log-linear baseline: distance curve, equipment, weight, market, quarter-end, trend."""

    name = "huber_log_linear"

    def fit(self, train: pd.DataFrame) -> "LinearModel":
        self.model = HuberRegressor(max_iter=2000, epsilon=1.35).fit(
            linear_design(train), np.log(train["posted_rate"])
        )
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(linear_design(frame))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(frame))


class GBMModel:
    """LightGBM on log rate. Trees cannot extrapolate calendar time, so `detrend`
    removes a linear time trend (estimated by the robust linear model) before
    fitting and adds it back when predicting."""

    def __init__(self, detrend: bool = True, features: list[str] | None = None, **overrides):
        self.detrend = detrend
        self.features = features or GBM_FEATURES
        self.params = {**GBM_PARAMS, **overrides}
        self.name = "lightgbm_detrended" if detrend else "lightgbm"

    def _trend(self, frame: pd.DataFrame) -> np.ndarray:
        return self.trend_per_year * frame["days_since_start"].to_numpy() / 365.0

    def fit(self, train: pd.DataFrame) -> "GBMModel":
        target = np.log(train["posted_rate"]).to_numpy()
        self.trend_per_year = 0.0
        if self.detrend:
            linear = LinearModel().fit(train)
            columns = list(linear_design(train.head(1)).columns)
            self.trend_per_year = float(linear.model.coef_[columns.index("trend_years")])
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(train[self.features], target - self._trend(train))
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame[self.features]) + self._trend(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(frame))


TIME_TERMS = ["trend_years", "quarter_end_ramp"]


class HybridModel:
    """log(rate) = GBM(load features) + b1 * trend + b2 * quarter_end_ramp.

    The GBM sees only load-level features (plus day of week), so it cannot memorise
    individual training dates; the calendar effects that must extrapolate into
    Nov-Dec are linear terms. The two parts are fitted by backfitting."""

    def __init__(self, features: list[str] | None = None, iterations: int = 2, **overrides):
        self.features = features or STATIC_FEATURES + ["day_of_week"]
        self.iterations = iterations
        self.params = {**GBM_PARAMS, **overrides}
        self.name = "hybrid_gbm_time_terms"

    def _offset(self, frame: pd.DataFrame) -> np.ndarray:
        return linear_design(frame)[TIME_TERMS].to_numpy() @ self.time_coef

    def fit(self, train: pd.DataFrame) -> "HybridModel":
        target = np.log(train["posted_rate"]).to_numpy()
        linear = LinearModel().fit(train)
        columns = list(linear_design(train.head(1)).columns)
        self.time_coef = np.array([linear.model.coef_[columns.index(term)] for term in TIME_TERMS])
        time_design = linear_design(train)[TIME_TERMS].to_numpy()
        for iteration in range(self.iterations):
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(train[self.features], target - self._offset(train))
            if iteration < self.iterations - 1:
                remainder = target - self.model.predict(train[self.features])
                refit = HuberRegressor(max_iter=2000, epsilon=1.35).fit(time_design, remainder)
                self.time_coef = refit.coef_
        return self

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(frame[self.features]) + self._offset(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(frame))


class BlendModel:
    """Geometric blend of the GBM and the linear model."""

    def __init__(self, gbm_weight: float = 0.8):
        self.gbm_weight = gbm_weight
        self.name = f"blend_{gbm_weight:.1f}gbm"

    def fit(self, train: pd.DataFrame) -> "BlendModel":
        self.gbm = GBMModel(detrend=True).fit(train)
        self.linear = LinearModel().fit(train)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        w = self.gbm_weight
        return np.exp(w * self.gbm.predict_log(frame) + (1 - w) * self.linear.predict_log(frame))
