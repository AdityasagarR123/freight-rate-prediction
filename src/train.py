"""Train the final model on all of data/train_test.csv and write both prediction files.

Usage: python src/train.py
Outputs:
  validation_predictions.csv          load_id,predicted_rate for the 12,000 validation loads
  data/december_chart_inputs.csv      predicted_rate column filled in place
  outputs/model.txt, outputs/model_meta.json, outputs/feature_importance.csv
  outputs/holdout_daily.png           Sep-Oct holdout: daily actual vs predicted
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import daily_market, load, prepare
from models import HybridModel, flag_label_outliers

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
HOLDOUT_START = "2025-09-01"


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            parts.append(frame[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def december_frame(december: pd.DataFrame, validation_raw: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """The December chart rows lack coordinates and market signals. Coordinates come from
    the city lookup; market_index / quote_signal are the daily means of the validation
    loads on the same date (the only market information available for those days)."""
    frame = december.drop(columns="predicted_rate").copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for side in ("pickup", "delivery"):
        frame[f"{side}_lat"] = frame[side].map(coords["lat"])
        frame[f"{side}_lon"] = frame[side].map(coords["lon"])
    daily = daily_market(validation_raw)
    frame["market_index"] = frame["date"].map(daily["day_market_index"])
    frame["quote_signal"] = frame["date"].map(daily["day_quote_signal"])
    return prepare(frame, daily)


def holdout_chart(model_cls, data: pd.DataFrame) -> dict[str, float]:
    """Refit on Jan-Aug, predict Sep-Oct, and plot the daily mean actual vs predicted."""
    train = data[data["date"] < HOLDOUT_START]
    test = data[data["date"] >= HOLDOUT_START].copy()
    model = model_cls().fit(train[~flag_label_outliers(train)])
    test["predicted"] = model.predict(test)
    test = test[~flag_label_outliers(train, test)]
    daily = test.groupby("date").apply(
        lambda g: pd.Series({"actual": np.exp(np.log(g["posted_rate"] / g["distance"]).mean()),
                             "predicted": np.exp(np.log(g["predicted"] / g["distance"]).mean())}),
        include_groups=False,
    )
    figure, axis = plt.subplots(figsize=(10, 4), dpi=150)
    axis.plot(daily.index, daily["actual"], color="#9DAFB3", linewidth=1.6, label="Actual (clean loads)")
    axis.plot(daily.index, daily["predicted"], color="#064A56", linewidth=2.2, label="Predicted")
    axis.set_title("Holdout Sep-Oct 2025: daily geometric-mean rate per mile", loc="left", fontweight="bold")
    axis.set_ylabel("$ / mile")
    axis.grid(axis="y", color="#D9E2E4")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUT / "holdout_daily.png")
    plt.close(figure)
    bias = float(np.mean(np.log(test["predicted"] / test["posted_rate"])))
    return {"holdout_log_bias": bias}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train_raw = load(DATA / "train_test.csv")
    validation_raw = load(DATA / "validation.csv")
    train = prepare(train_raw)
    validation = prepare(validation_raw)

    outliers = flag_label_outliers(train)
    model = HybridModel().fit(train[~outliers])
    print(f"Trained on {(~outliers).sum():,} loads ({outliers.sum()} label outliers removed).")

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    predictions = pd.Series(model.predict(validation), index=validation["load_id"])
    template["predicted_rate"] = template["load_id"].map(predictions).round(2)
    assert template["predicted_rate"].notna().all() and len(template) == 12_000
    template.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv (mean {template['predicted_rate'].mean():.2f}).")

    december_path = DATA / "december_chart_inputs.csv"
    december = pd.read_csv(december_path)
    coords = city_coordinates(train_raw, validation_raw)
    december["predicted_rate"] = model.predict(december_frame(december, validation_raw, coords)).round(2)
    december.to_csv(december_path, index=False)
    print("December predictions:", december["predicted_rate"].describe().round(2).to_dict())

    model.model.booster_.save_model(str(OUT / "model.txt"))
    importance = pd.Series(
        model.model.booster_.feature_importance("gain"), index=model.features, name="gain"
    ).sort_values(ascending=False)
    (importance / importance.sum()).round(4).to_csv(OUT / "feature_importance.csv")
    meta = {
        "model": model.name,
        "features": model.features,
        "time_coefficients": dict(zip(["trend_per_year", "quarter_end_ramp"], model.time_coef.round(5).tolist())),
        "train_rows_used": int((~outliers).sum()),
        "label_outliers_removed": int(outliers.sum()),
        **holdout_chart(HybridModel, train),
    }
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
