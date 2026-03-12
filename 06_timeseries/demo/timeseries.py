import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# --- Output directory
OUT_DIR = "outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(filename):
    """Save current figure to outputs/figures/ and close it."""
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR}/{filename}")

# --- 0) Setup
np.random.seed(123)

# --- 1) Create a synthetic daily time series (trend + weekly seasonality + AR(1) noise)
n = 600
dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
t = np.arange(1, n + 1)

trend = 0.02 * t
weekly = 1.2 * np.sin(2 * np.pi * t / 7)

phi = 0.65
eps = np.random.normal(loc=0.0, scale=1.0, size=n)
ar_noise = np.empty(n)
ar_noise[0] = eps[0]
for i in range(1, n):
  ar_noise[i] = phi * ar_noise[i - 1] + eps[i]

y = 10 + trend + weekly + ar_noise

df = pd.DataFrame({"date": dates, "t": t, "y": y})

# --- 2) Visualize the series
plt.figure()
plt.plot(df["date"], df["y"])
plt.title("Synthetic daily time series: trend + weekly seasonality + AR(1) noise")
plt.xlabel("Date")
plt.ylabel("y")
plt.tight_layout()
savefig("01_synthetic_series.png")

############################################
# PART A: Time leakage demo (random split vs time split)
############################################

# --- 3) WRONG evaluation: random train/test split (time leakage)
np.random.seed(123)
test_frac = 0.20
test_n = int(np.floor(n * test_frac))

all_idx = np.arange(n)
test_idx_random = np.random.choice(all_idx, size=test_n, replace=False)
train_idx_random = np.setdiff1d(all_idx, test_idx_random)

y_train_random = df.loc[train_idx_random, "y"].to_numpy()
y_test_random = df.loc[test_idx_random, "y"].to_numpy()

# Fit ARIMA(1,0,0) on randomly selected training points (conceptually wrong for time series)
fit_random = ARIMA(y_train_random, order=(1, 0, 0)).fit()
pred_random = fit_random.forecast(steps=len(y_test_random))

rmse_random = np.sqrt(np.mean((y_test_random - pred_random) ** 2))

print("\n==============================")
print(f"WRONG: Random split RMSE (time leakage): {rmse_random:.6f}")
print("==============================")

# --- 4) RIGHT evaluation: train on past, test on future
cut = n - test_n
train_idx_time = np.arange(0, cut)
test_idx_time = np.arange(cut, n)

y_train_time = df.loc[train_idx_time, "y"].to_numpy()
y_test_time = df.loc[test_idx_time, "y"].to_numpy()

fit_time = ARIMA(y_train_time, order=(1, 0, 0)).fit()
pred_time = fit_time.forecast(steps=len(y_test_time))

rmse_time = np.sqrt(np.mean((y_test_time - pred_time) ** 2))

print("\n==============================")
print(f"RIGHT: Time split RMSE (train past, test future): {rmse_time:.6f}")
print("==============================")

# --- 5) Plot the correct evaluation (train vs test and forecast)
plt.figure()
plt.plot(df["date"], df["y"], label="Observed y")
plt.axvline(df.loc[cut, "date"], linestyle="--", label="Train/Test cutoff")
plt.plot(df.loc[test_idx_time, "date"], pred_time, label="Forecast (future)")
plt.title("Correct evaluation: train on past, test on future")
plt.xlabel("Date")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
savefig("02_time_split_forecast.png")

############################################
# PART B: Synthetic DGP demo (autocorrelation + trend) + ACF/PACF diagnostics
############################################

# --- 6) Generate data from a known DGP: trend + AR(1) errors
# DGP: y_t = alpha + delta*t + e_t ; e_t = phi*e_{t-1} + u_t
np.random.seed(123)
n2 = 300
t2 = np.arange(1, n2 + 1)

alpha = 5
delta = 0.03
phi2 = 0.75
u = np.random.normal(loc=0.0, scale=1.0, size=n2)

e = np.empty(n2)
e[0] = u[0]
for i in range(1, n2):
  e[i] = phi2 * e[i - 1] + u[i]

y2 = alpha + delta * t2 + e

# --- 7) Plot the DGP series
plt.figure()
plt.plot(t2, y2)
plt.title("Synthetic DGP: linear trend + AR(1) errors")
plt.xlabel("t")
plt.ylabel("y_t")
plt.tight_layout()
savefig("03_dgp_series.png")

# --- 8) Diagnose dependence with ACF and PACF
plt.figure()
plot_acf(y2, ax=plt.gca(), lags=40)
plt.title("ACF of y_t (trend + AR errors)")
plt.tight_layout()
savefig("04_acf_y2.png")

plt.figure()
plot_pacf(y2, ax=plt.gca(), lags=40, method="ywm")
plt.title("PACF of y_t (trend + AR errors)")
plt.tight_layout()
savefig("05_pacf_y2.png")

# --- 9) Detrend and re-check ACF/PACF on residuals
# Remove linear trend via OLS: y2 ~ 1 + t2
X = np.column_stack([np.ones(n2), t2])
beta_hat = np.linalg.lstsq(X, y2, rcond=None)[0]
y2_hat = X @ beta_hat
resid2 = y2 - y2_hat

plt.figure()
plt.plot(t2, resid2)
plt.title("Residuals after removing linear trend (should still show AR structure)")
plt.xlabel("t")
plt.ylabel("residual")
plt.tight_layout()
savefig("06_detrended_residuals.png")

plt.figure()
plot_acf(resid2, ax=plt.gca(), lags=40)
plt.title("ACF of residuals (trend removed)")
plt.tight_layout()
savefig("07_acf_residuals.png")

plt.figure()
plot_pacf(resid2, ax=plt.gca(), lags=40, method="ywm")
plt.title("PACF of residuals (trend removed)")
plt.tight_layout()
savefig("08_pacf_residuals.png")

# --- 10) Fit an AR(1) model to residuals and compare estimated phi to truth
fit_ar1 = ARIMA(resid2, order=(1, 0, 0)).fit()
phi_hat = fit_ar1.params[1]  # AR1 coefficient (params[0] is intercept by default)

print("\n==============================")
print(f"DGP truth phi2 = {phi2}")
print(f"Estimated AR(1) phi from residuals = {phi_hat:.6f}")
print("==============================")

# --- 11) Narration-ready takeaway
print("\nNarration-ready takeaway:")
print("- In the DGP, we *know* the errors are AR(1), so observations are dependent over time.")
print("- ACF/PACF make that dependence visible.")
print("- Removing trend helps isolate autocorrelation in the error process.")
print("- Separately: random splits leak time and look too good; past->future splits are the honest default.")



###############################################################################
# Time Series HW — Question 4 and Question 6
# Requires: pip install statsmodels pandas numpy matplotlib
###############################################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA

# --- Output directories
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables",  exist_ok=True)

def savefig(filename):
    plt.savefig(os.path.join("outputs/figures", filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: outputs/figures/{filename}")

# --- Reproduce the same synthetic series from the tutorial
np.random.seed(123)
n = 600
dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
t = np.arange(1, n + 1)

trend   = 0.02 * t
weekly  = 1.2 * np.sin(2 * np.pi * t / 7)
phi     = 0.65
eps     = np.random.normal(0, 1, n)
ar_noise = np.empty(n)
ar_noise[0] = eps[0]
for i in range(1, n):
    ar_noise[i] = phi * ar_noise[i - 1] + eps[i]

y = 10 + trend + weekly + ar_noise
df = pd.DataFrame({"date": dates, "t": t, "y": y})


###############################################################################
# QUESTION 4: Decomposition — trend + seasonality + residual
###############################################################################

# STL (Seasonal-Trend decomposition using LOESS) requires a pandas Series
# with a DatetimeIndex and a declared period.
series = pd.Series(y, index=dates, name="y")

# period=7 tells STL the seasonal cycle is 7 days (weekly)
stl = STL(series, period=7, robust=True)
result = stl.fit()

# result gives us three components:
#   result.trend    — smooth long-run level
#   result.seasonal — repeating weekly pattern
#   result.resid    — leftover noise

fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

axes[0].plot(dates, y,               color="#333333", linewidth=0.8)
axes[0].set_ylabel("Observed")
axes[0].set_title("STL Decomposition — Synthetic Daily Series (period = 7)")

axes[1].plot(dates, result.trend,    color="#2166ac", linewidth=1.2)
axes[1].set_ylabel("Trend")

axes[2].plot(dates, result.seasonal, color="#4dac26", linewidth=0.8)
axes[2].set_ylabel("Seasonal")

axes[3].plot(dates, result.resid,    color="#d6604d", linewidth=0.6, alpha=0.8)
axes[3].axhline(0, color="black", linewidth=0.7, linestyle="--")
axes[3].set_ylabel("Residual")
axes[3].set_xlabel("Date")

plt.tight_layout()
savefig("decomposition.png")


###############################################################################
# QUESTION 6: Interrupted Time Series (ITS) + Placebo
###############################################################################

# --- Choose intervention date: t0 = 300 (middle of the 600-day series)
t0 = 300

# --- ITS design variables
# D_t   = 1 if t >= t0 (level shift indicator)
# P_t   = (t - t0) * D_t (post-intervention time counter for slope change)
df["D"]  = (df["t"] >= t0).astype(int)
df["P"]  = (df["t"] - t0) * df["D"]

# --- Fit the real ITS model:
#   y_t = alpha + delta*t + tau1*D_t + tau2*P_t + e_t
#
#   alpha         : baseline intercept
#   delta         : pre-existing trend slope
#   tau1          : immediate level change at t0
#   tau2          : change in slope after t0

from statsmodels.formula.api import ols as smf_ols

its_model = smf_ols("y ~ t + D + P", data=df).fit()
df["fitted_its"] = its_model.fittedvalues

# Counterfactual: what would have happened without the intervention?
# Set tau1 = tau2 = 0 — i.e., use only alpha + delta*t
df["counterfactual"] = its_model.params["Intercept"] + its_model.params["t"] * df["t"]

print("\nReal ITS coefficients:")
print(its_model.summary().tables[1])

# --- Placebo ITS: fake intervention at t0_placebo = 150 (pre-period)
t0_placebo = 150
df["D_p"] = (df["t"] >= t0_placebo).astype(int)
df["P_p"] = (df["t"] - t0_placebo) * df["D_p"]

placebo_model = smf_ols("y ~ t + D_p + P_p", data=df).fit()

print("\nPlacebo ITS coefficients (fake t0 = 150):")
print(placebo_model.summary().tables[1])

# --- Save coefficient tables
def model_to_df(model, label):
    tbl = model.summary2().tables[1].copy()
    tbl.insert(0, "model", label)
    tbl.index.name = "term"
    return tbl.reset_index()

coef_df = pd.concat([
    model_to_df(its_model,     "real_ITS"),
    model_to_df(placebo_model, "placebo_ITS"),
], ignore_index=True)
coef_df.to_csv("outputs/tables/its_results.csv", index=False)
print("Saved: outputs/tables/its_results.csv")

# --- ITS figure
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

# Top panel: real ITS
ax = axes[0]
ax.plot(df["t"], df["y"],             color="#aaaaaa", linewidth=0.7, label="Observed $y_t$", zorder=1)
ax.plot(df["t"], df["fitted_its"],    color="#2166ac", linewidth=1.6, label="ITS fitted",     zorder=3)
ax.plot(df["t"], df["counterfactual"],color="#d6604d", linewidth=1.4,
        linestyle="--", label="Counterfactual (no intervention)",      zorder=2)
ax.axvline(t0, color="black", linewidth=1.2, linestyle=":", label=f"Intervention $t_0={t0}$")
ax.set_xlabel("t")
ax.set_ylabel("$y_t$")
ax.set_title(f"Real ITS — intervention at $t_0 = {t0}$\n"
             rf"$\hat\tau_1={its_model.params['D']:.3f}$ (level),  "
             rf"$\hat\tau_2={its_model.params['P']:.4f}$ (slope)")
ax.legend(fontsize=8)

# Bottom panel: placebo ITS (pre-period fake date)
df["fitted_placebo"] = placebo_model.fittedvalues
ax2 = axes[1]
ax2.plot(df["t"], df["y"],              color="#aaaaaa", linewidth=0.7, label="Observed $y_t$", zorder=1)
ax2.plot(df["t"], df["fitted_placebo"], color="#74c476", linewidth=1.6, label="Placebo fitted",  zorder=3)
ax2.axvline(t0_placebo, color="black", linewidth=1.2, linestyle=":",
            label=f"Fake intervention $t_0={t0_placebo}$")
ax2.set_xlabel("t")
ax2.set_ylabel("$y_t$")
tau1_p = placebo_model.params["D_p"]
tau2_p = placebo_model.params["P_p"]
ax2.set_title(f"Placebo ITS — fake intervention at $t_0 = {t0_placebo}$ (pre-period)\n"
              rf"$\hat\tau_1={tau1_p:.3f}$ (level),  $\hat\tau_2={tau2_p:.4f}$ (slope)  "
              "— should be ≈ 0")
ax2.legend(fontsize=8)

plt.tight_layout()
savefig("its_plot.png")

print("\nAll outputs saved.")