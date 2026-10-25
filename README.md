# Freight Rate Prediction

Predicts `posted_rate` for truckload freight. The model trains on Jan–Oct 2025 loads and forecasts the Nov–Dec 2025 validation loads.

**Model:** LightGBM on `log(rate)`, using load-level features (distance, equipment, weight, lat/lon, market index, quote signal, weekday), plus two linear time terms (a long-run trend and a quarter-end ramp) that extrapolate into unseen months. The two parts are fitted by robust backfitting. Rows with corrupted labels are removed before training.

**Out-of-time backtest** (three expanding-window folds, each with a 2-month test window):

| Model | Clean MAPE | Clean MAE | All-rows MAPE |
|---|---|---|---|
| Hybrid GBM + time terms (chosen) | 2.13% | $51.8 | 4.49% |
| LightGBM with calendar features, detrended | 2.21% | $52.7 | 4.58% |
| LightGBM with calendar features | 2.73% | $62.2 | 5.02% |
| Huber log-linear baseline | 2.81% | $60.7 | 5.17% |

"Clean" leaves out the ~1.4% of test labels that are corrupted (0.2–0.5x or 2–6x the expected rate). The report has the full write-up: [`report/Freight_Rate_Report.docx`](report/Freight_Rate_Report.docx).

## Layout

```
data/                     input CSVs (december_chart_inputs.csv is filled in by train.py)
src/features.py           cleaning + feature engineering (shared by training and inference)
src/models.py             outlier filter, baseline, GBM variants, final hybrid model
src/validate.py           out-of-time backtest -> outputs/cv_results.csv
src/train.py              final fit, validation_predictions.csv, December predictions
src/make_report.py        builds report/Freight_Rate_Report.docx
score.py                  provided scorer (validates outputs, draws December chart)
validation_predictions.csv
scorer_results/candidate_december.png
```

## Run

Python 3.10+ is required. The pipeline was developed with Python 3.14.

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

python src/validate.py        # backtest (about 2 min)
python src/train.py           # final model + both prediction files
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python src/make_report.py     # report/Freight_Rate_Report.docx
```

## Data-quality handling

| Issue | Handling |
|---|---|
| Corrupted `posted_rate` labels (~1.4%) | Dropped when the absolute log residual from a robust Huber fit exceeds 0.25 |
| Negative weights | Sign error: replaced with `abs(weight)` |
| Missing weight | Equipment median, plus a missing-value flag |
| Missing `market_index` | Same-day mean (the index is effectively a daily series) |
| 8 cities appear only in validation | No city IDs as features; location enters through lat/lon |
| Distance disagrees with coordinates | Kept: coordinates are jittered and distance has a 70-mile floor. Rates follow posted distance |

December chart rows have no `market_index` or `quote_signal`. Each date uses the daily mean of the validation loads on that date.
