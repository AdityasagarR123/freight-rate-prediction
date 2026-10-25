"""Time-based backtesting of the candidate models on data/train_test.csv.

Usage: python src/validate.py
Writes outputs/cv_results.csv and outputs/cv_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import load, prepare
from models import GBMModel, HybridModel, LinearModel, flag_label_outliers

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Expanding-window folds with 2-month test windows, mirroring the real task:
# train on everything up to a cutoff, predict the next two months (validation = Nov-Dec).
FOLDS = [
    ("F1", "2025-05-01", "2025-07-01"),
    ("F2", "2025-07-01", "2025-09-01"),
    ("F3", "2025-09-01", "2025-11-01"),
]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    ape = np.abs(error) / actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(ape) * 100),
        "mdape": float(np.median(ape) * 100),
    }


def candidates():
    return [
        LinearModel(),
        GBMModel(detrend=False),
        GBMModel(detrend=True),
        HybridModel(),
    ]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = load(ROOT / "data" / "train_test.csv")
    data = prepare(raw)
    rows = []
    for fold, start, end in FOLDS:
        train = data[data["date"] < start].reset_index(drop=True)
        test = data[(data["date"] >= start) & (data["date"] < end)].reset_index(drop=True)
        outliers_train = flag_label_outliers(train)
        # Test outliers are flagged with a filter fitted on the *training* fold only.
        outliers_test = flag_label_outliers(train, test)
        clean_train = train[~outliers_train]
        print(f"{fold}: train {len(train):,} (dropped {outliers_train.sum()} outliers), test {len(test):,}")
        for model in candidates():
            for train_set, label in [(clean_train, "clean"), (train, "raw")]:
                if label == "raw" and not isinstance(model, HybridModel):
                    continue
                fitted = model.fit(train_set)
                predicted = fitted.predict(test)
                actual = test["posted_rate"].to_numpy()
                name = model.name + ("" if label == "clean" else "_no_outlier_filter")
                all_rows = metrics(actual, predicted)
                clean_rows = metrics(actual[~outliers_test], predicted[~outliers_test])
                rows.append({"fold": fold, "model": name, **{f"all_{k}": v for k, v in all_rows.items()},
                             **{f"clean_{k}": v for k, v in clean_rows.items()}})
                print(f"  {name:38s} all MAE {all_rows['mae']:7.2f} MAPE {all_rows['mape']:5.2f}% | "
                      f"clean MAE {clean_rows['mae']:6.2f} MAPE {clean_rows['mape']:5.2f}%")
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "cv_results.csv", index=False)
    summary = results.drop(columns="fold").groupby("model").mean().sort_values("clean_mae")
    print(summary.round(2).to_string())
    summary.round(3).to_json(OUT / "cv_summary.json", orient="index", indent=2)


if __name__ == "__main__":
    main()
