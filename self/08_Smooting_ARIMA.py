# ---------- Q8: CO2 forecasting with smoothing and ARIMA ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the weekly Mauna Loa CO2 data
co2 = sm.datasets.co2.load_pandas().data
print(co2.head())
print("Missing values:", co2["co2"].isna().sum())

# 2. Convert to a regular monthly series and fill gaps
monthly = co2["co2"].resample("MS").mean().interpolate()
print("\nMonthly series shape:", monthly.shape)

plt.figure(figsize=(10, 3))
monthly.plot(title="Monthly CO2 (Mauna Loa)")
plt.ylabel("CO2 (ppm)")
plt.show()

# 3. Chronological split (never shuffle time series)
split = int(len(monthly) * 0.8)
train, test = monthly[:split], monthly[split:]
print("Train:", train.index.min().date(), "->", train.index.max().date())
print("Test :", test.index.min().date(), "->", test.index.max().date())

# 4. Baseline + smoothing models
forecasts = {}

# (a) moving-average baseline: last 12-month mean, repeated
ma_value = train.rolling(window=12).mean().iloc[-1]
forecasts["Moving Average"] = pd.Series(ma_value, index=test.index)

# (b) simple exponential smoothing (level only)
ses = SimpleExpSmoothing(train, initialization_method="estimated").fit()
forecasts["SES"] = pd.Series(np.asarray(ses.forecast(len(test))), index=test.index)

# (c) Holt's linear trend
holt = Holt(train, initialization_method="estimated").fit()
forecasts["Holt"] = pd.Series(np.asarray(holt.forecast(len(test))), index=test.index)

# (d) Holt-Winters (additive trend + seasonality, period = 12 months)
hw = ExponentialSmoothing(train, trend="add", seasonal="add",
                          seasonal_periods=12,
                          initialization_method="estimated").fit()
forecasts["Holt-Winters"] = pd.Series(np.asarray(hw.forecast(len(test))), index=test.index)

# 5. ACF / PACF on the differenced series to motivate AR/MA orders
diff_train = train.diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
plot_acf(diff_train, lags=24, ax=axes[0]); axes[0].set_title("ACF (differenced)")
plot_pacf(diff_train, lags=24, ax=axes[1]); axes[1].set_title("PACF (differenced)")
plt.tight_layout(); plt.show()

# 6. ARIMA with d=1 (integration removes the trend)
arima = ARIMA(train, order=(1, 1, 1)).fit()
forecasts["ARIMA(1,1,1)"] = pd.Series(np.asarray(arima.forecast(len(test))), index=test.index)

# 7. Compare error metrics
rows = []
for name, fc in forecasts.items():
    rows.append({"Method": name,
                 "MAE": mean_absolute_error(test, fc),
                 "RMSE": np.sqrt(mean_squared_error(test, fc))})
print("\nForecast errors:\n", pd.DataFrame(rows).set_index("Method").round(3))

# 8. Plot actual vs forecasts
plt.figure(figsize=(11, 4))
plt.plot(train.index[-60:], train.iloc[-60:], label="Train (tail)")
plt.plot(test.index, test, label="Actual", color="black", linewidth=2)
for name, fc in forecasts.items():
    plt.plot(test.index, fc, label=name)
plt.legend(); plt.title("CO2 forecasts on the held-out period")
plt.ylabel("CO2 (ppm)")
plt.show()
