# Feature testing results (`test_ppt`, 10/29)

> **Note:** these runs predicted the **daily mean** of `SWC_20` on the day of the hour 72 h after the window,
> not the hourly value at that hour. The follow-up `combined_10yr.ipynb` uses the hourly target, so its numbers
> are not directly comparable with the ones below.

## Overview

- **Goal:** find which single feature, added to a minimal base, most improves the 72h forecast of FD11 `SWC_20`,
  and which rain window the base should use.
- **Notebooks:** `feature_testing_ppt3h`, `_ppt6h` and `_ppt24h`. They are identical except for the base rain
  input, `Ppt_sum_log_{3,6,24}h`: `log1p` of the rain over the last 3, 6 or 24 h, after setting hourly rain below
  1 mm to 0.
- **Runs:** each notebook trains the base (`SWC_20` + its rain total), then 21 runs that each add one candidate,
  so 66 train/test runs in total. All ran on a Colab GPU in about 5 minutes per notebook.
- **Setup** (same as `lstm_72_new`):
  - Data: train 2019–2021, validation 2022 for early stopping, test 2024. 2023 is skipped because of the rain gauge
    outage.
  - Window and target: 48h input window. The target is the daily mean of `SWC_20` 72 h ahead, predicted as a change
    from the last value, with Huber loss.
  - Model and training: `model_test`'s BiLSTM, Adam lr 1e-3, batch 32, stride 8, up to 30 epochs with patience 10,
    seed 42.
- **Skill** = 1 − model / last value, on the same windows; positive means the model beats persistence.

  | Last value | RMSE | MAE |
  |---|---|---|
  | Validation 2022 | 0.03167 | 0.01620 |
  | Test 2024 | 0.03227 | 0.01741 |

## Findings

### Rain window: 3h, 6h and 24h perform about the same

| Base | val RMSE | val MAE | test RMSE | test MAE | test bias |
|---|---|---|---|---|---|
| `SWC_20` + `Ppt_sum_log_3h` | +0.2% | +10.6% | +1.3% | +14.2% | −0.0048 |
| `SWC_20` + `Ppt_sum_log_6h` | +1.1% | +10.3% | +1.7% | **+15.4%** | −0.0040 |
| `SWC_20` + `Ppt_sum_log_24h` | **+1.2%** | +10.6% | **+2.3%** | +14.9% | −0.0038 |

- All three bases beat last value, and they sit within about 1 point of each other, which is inside run-to-run
  noise. The 24h total is slightly best on RMSE in both years.
- Adding longer rain memory (72h or 168h totals, hours since rain, decaying rain index) does nothing in any
  notebook.

### Which features worked best

Validation MAE skill / test MAE skill in each notebook. The base row is for reference.

| Feature added | 3h | 6h | 24h | Consistent? |
|---|---|---|---|---|
| `Tair` (model-filled at FD11) | +21.9 / +26.1 | +22.2 / **+27.8** | +21.9 / +26.9 | yes, top on MAE everywhere |
| `T_5` | +18.0 / +20.6 | +19.8 / +23.9 | +21.4 / +24.7 | yes |
| `T_5_mean_24h` | +18.9 / +24.4 | +19.8 / +23.7 | +20.6 / +23.0 | yes, best measured, best val RMSE in 6h/24h |
| `T_20` | +16.9 / +22.7 | +17.5 / +21.9 | +19.3 / +23.6 | yes on MAE, but best epoch 3–7 and weak RMSE |
| `SWC_20_below_max_72h` | +16.0 / +18.4 | +16.4 / +19.6 | +17.8 / +19.2 | yes |
| `day_of_year` | +15.6 / +19.0 | +15.4 / +20.8 | +16.2 / +22.5 | yes |
| `season` | +15.1 / +21.4 | +13.4 / +20.7 | +13.3 / +19.5 | yes, a little weaker than `day_of_year` on validation |
| `SWC_5` | +13.2 / +19.0 | +12.9 / +16.3 | +13.1 / +17.5 | yes, **best test RMSE** in all three (+3.1% to +3.6%) |
| `SWC_20_change_72h` | +16.0 / +19.7 | +10.9 / +14.9 | +10.6 / +16.0 | only with the 3h base |
| **base** | +10.6 / +14.2 | +10.3 / +15.4 | +10.6 / +14.9 | |

1. **Temperature helps most.** `Tair`, `T_5`, `T_5_mean_24h` and `T_20` are the top four in every notebook, adding
   +6 to +11 points of MAE skill over the base. With only `SWC_20` and rain, temperature is the model's only signal
   for how fast the soil is drying.
   - `T_5_mean_24h` and `T_5` are the measured choices. `Tair` scores slightly better on MAE, but it is 100%
     model-filled at FD11.
2. **Time of year helps** (`day_of_year`, `season`; +3 to +7 points). In 2019–2021, day of year explains 87% of the
   variance of `T_5_mean_24h` (91% for `T_20`), so the two carry largely the same seasonal signal. No run has
   tested both together yet, so whether the model gains anything from having both is unknown.
3. **`SWC_20_below_max_72h` helps in every notebook** (+5 to +7 points on validation). It tells the model how far
   the soil has drained since its recent peak.
4. **`SWC_5` helps RMSE most.** It has the best test RMSE skill in all three notebooks, with moderate MAE gains.
5. **No effect in any notebook** (within about 1.5 points of the base on validation MAE):
   - the 24h means and changes of `SWC_20` and `SWC_5`
   - `SWC_20_above_min_30d`, hour of day, wind
   - the 72h and 168h rain totals, hours since rain, decaying rain index

   The model can already work out the 24h `SWC` features from the 48 hours of raw values it sees.
6. **`T_5_range_24h` hurts on test:** 2.2–2.6 points below the base in all three notebooks.

**How big a gap is real:** the features the model can already compute (the 24h `SWC` changes and means) land within
1 point of the base on validation MAE and within about 2 points on test. Treat that as the single-seed noise floor:
the temperature, time-of-year and `SWC_20_below_max_72h` gains are well above it, but the differences between the
top four temperature features are not.

### Important metric comparisons

- **Every one of the 66 runs beats last value on MAE:** validation +9.8% to +22.2%, test +11.6% to +27.8%.
- **RMSE gains are small everywhere:** validation −0.4% to +3.5%, test −0.5% to +3.6%. Rain that falls during the
  72h gap can't be seen from past data, and it dominates RMSE.
- **Best absolute test scores:**

  | | Test RMSE | Test MAE |
  |---|---|---|
  | Best MAE: `Ppt_sum_log_6h` + `Tair` | 0.03189 | **0.01257** (+27.8%) |
  | Best measured MAE: `Ppt_sum_log_24h` + `T_5` | 0.03164 | 0.01312 (+24.7%) |
  | Best RMSE: `Ppt_sum_log_3h` + `SWC_5` | **0.03109** (+3.6%) | 0.01411 |
  | Last value | 0.03227 | 0.01741 |

- **Every model is too dry:** test bias is −0.0027 to −0.0069 in every run. The Huber loss aims at the typical
  no-rain outcome, so the models under-predict when rain does come.
- **Training was cut short.** Stride 8 leaves 3,268 training windows, about 100 batches per epoch. 33 of the 66
  runs used all 30 epochs, and 24 had their best epoch at 26–30, so many were still improving when the limit
  stopped them.
- **Against the other notebooks (test 2024):**

| Notebook | Target | Test RMSE skill | Test MAE skill |
|---|---|---|---|
| `akhil_lstm_72h` 10/29 run (11 features, 1-layer LSTM, stride 1) | hourly `SWC_20` | +1.4% | +23.0% |
| `lstm_72_new` with `model_test`'s optimizer (local check, before the fix) | daily mean | −0.4% | +7.0% |
| `test_ppt`, best base (`Ppt_sum_log_24h`) | daily mean | +2.3% | +14.9% |
| `test_ppt`, `Ppt_sum_log_24h` + `T_5` | daily mean | +2.0% | +24.7% |

`akhil_lstm_72h` predicts the hourly value, so its skill is measured against a different last-value baseline (RMSE
0.03345, MAE 0.01779). With only 3 inputs, one temperature feature matches its MAE skill.

## Next steps

1. **Use `Ppt_sum_log_24h` as the base rain input.** The windows tie, and 24h is best on RMSE in both years.
2. **Test the winners together.** In `feature_testing_ppt24h`, add one entry to `FEATURE_SETS`:
   `"combined": BASE_FEATURES + ["T_5_mean_24h", "doy_sin", "doy_cos", "SWC_20_below_max_72h", "SWC_5"]`.
   This checks whether the gains add up.
3. **Train longer.** Set `TRAINING_STRIDE = 1` (8× more training windows, about 40 minutes per notebook on a GPU).
   Epochs and patience stay at 30 and 10.
4. **Measure the noise.** Rerun `base`, `+ T_5_mean_24h` and `combined` with `SEED` set to 0, 1 and 2.
5. **Drop the no-effect features from future runs:** the 24h `SWC` means and changes, `SWC_20_above_min_30d`,
   `T_5_range_24h`, hour of day, wind, and the extra rain-memory features.
