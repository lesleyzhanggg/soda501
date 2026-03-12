###############################################################################
# API Use + Forecasting Tutorial: Python
# Author: Jared Edgerton
# Date: date.today()
#
# This script demonstrates:
#   1) Loading and cleaning presidential vote data (1976–2020)
#   2) Pulling economic indicators from FRED (Q1/Q2 of election years)
#   3) Building a simple national vote-share model (OLS)
#   4) Loading state-level poll + census data and fitting a state model (OLS)
#   5) Producing a simple 2020 state-level visualization
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - You will see the same steps repeated so students can follow the logic and
#   edit one piece at a time.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
# If you do not have these installed, run (in Terminal / Anaconda Prompt):
#   pip install pandas numpy matplotlib statsmodels fredapi pyreadr plotly lxml requests

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import date
import statsmodels.formula.api as smf

# FRED API wrapper
from fredapi import Fred

# For reading .rds (RDS) files in Python (state-level poll/census data)
import pyreadr

# For a quick US states choropleth
import plotly.express as px

import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()




# -----------------------------------------------------------------------------
# Part 1: Presidential vote data (national-level)
# -----------------------------------------------------------------------------
# Read in the presidential election vote data
vote_data = pd.read_csv("1976-2020-president.csv")

# Keep only Democrat and Republican votes
vote_data = vote_data[
    vote_data["party_detailed"].isin(["DEMOCRAT", "REPUBLICAN"])
].copy()

# Summarize votes by year, candidate, party (mimics ddply summarize in R)
vote_data = (
    vote_data
    .groupby(["year", "candidate", "party_detailed"], as_index=False)
    .agg(
        candidatevotes=("candidatevotes", "sum"),
        totalvotes=("totalvotes", "sum")
    )
)

# Drop OTHER and blank candidate entries (mimics R filters)
vote_data = vote_data[
    (~vote_data["candidate"].isin(["OTHER", ""])) &
    (vote_data["candidate"].notna())
].copy()

# Compute vote percent
vote_data["vote_pct"] = vote_data["candidatevotes"] / vote_data["totalvotes"]

# Election years used in this dataset
election_years = np.sort(vote_data["year"].unique())


# -----------------------------------------------------------------------------
# Part 2: Pulling economic indicators from FRED (Q1/Q2 of election years)
# -----------------------------------------------------------------------------
# NOTE: Replace with your own key (students should get one from FRED).
fred_api_key = "2ea96e92747bd457e761a6be7931daea"
fred = Fred(api_key=fred_api_key)

# Define observation window based on the election years in the vote data
obs_start = f"{int(election_years.min())}-01-01"
obs_end   = f"{int(election_years.max())}-06-30"

# --- Unemployment (UNRATE) ---
# FRED returns a time series with dates; we convert to quarterly and keep Q1/Q2


unrate = fred.get_series("UNRATE", observation_start=obs_start, observation_end=obs_end)
unrate = unrate.to_frame(name="unemployment_rate")
unrate.index = pd.to_datetime(unrate.index)
unrate = unrate.resample("QE").mean().reset_index().rename(columns={"index": "date"})
unrate["year"] = unrate["date"].dt.year
unrate["quarter"] = unrate["date"].dt.quarter
unemployment_data = unrate[
    (unrate["year"].isin(election_years)) &
    (unrate["quarter"] <= 2)
][["year", "quarter", "unemployment_rate"]].copy()

# --- GDP (GDP) ---
gdp = fred.get_series("GDP", observation_start=obs_start, observation_end=obs_end)
gdp = gdp.to_frame(name="gdp")
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.resample("QE").mean().reset_index().rename(columns={"index": "date"})
gdp["year"] = gdp["date"].dt.year
gdp["quarter"] = gdp["date"].dt.quarter
gdp_data = gdp[
    (gdp["year"].isin(election_years)) &
    (gdp["quarter"] <= 2)
][["year", "quarter", "gdp"]].copy()

# --- CPI (CPIAUCSL) ---
cpi = fred.get_series("CPIAUCSL", observation_start=obs_start, observation_end=obs_end)
cpi = cpi.to_frame(name="cpi")
cpi.index = pd.to_datetime(cpi.index)
cpi = cpi.resample("QE").mean().reset_index().rename(columns={"index": "date"})
cpi["year"] = cpi["date"].dt.year
cpi["quarter"] = cpi["date"].dt.quarter
cpi_data = cpi[
    (cpi["year"].isin(election_years)) &
    (cpi["quarter"] <= 2)
][["year", "quarter", "cpi"]].copy()

# (Optional, for teaching) inflation rate example (year-over-year using Q1 vs Q3 lag etc.)
# The original R code computed inflation_rate and then dropped it before widening.
# We replicate the same idea but do not use it in the final wide dataset.
inflation_data = cpi_data.sort_values(["year", "quarter"]).copy()
inflation_data["inflation_rate"] = (
    (inflation_data["cpi"] / inflation_data["cpi"].shift(2) - 1) * 100
)

# Combine all economic data into one long table keyed by (year, quarter)
combined_long = (
    unemployment_data
    .merge(gdp_data, on=["year", "quarter"], how="outer")
    .merge(inflation_data[["year", "quarter", "cpi"]], on=["year", "quarter"], how="outer")
    .sort_values(["year", "quarter"])
)

# Pivot wider like R pivot_wider(names_from=quarter, values_from=c(...), names_sep="_Q")
combined_wide = combined_long.pivot_table(
    index="year",
    columns="quarter",
    values=["unemployment_rate", "gdp", "cpi"],
    aggfunc="first"
)

# Flatten column names to match the R naming style, e.g. unemployment_rate_Q1
combined_wide.columns = [f"{var}_Q{q}" for var, q in combined_wide.columns]
combined_wide = combined_wide.reset_index()


# -----------------------------------------------------------------------------
# Part 3: Merge vote data + economic data and build national forecast features
# -----------------------------------------------------------------------------
forecast_data = vote_data.merge(combined_wide, on="year", how="left").copy()

# Incumbent indicator (hard-coded, sequential assignments like the R mutate/ifelse chain)
forecast_data["incumbent"] = 0
forecast_data.loc[(forecast_data["candidate"] == "FORD, GERALD") & (forecast_data["year"] == 1976), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CARTER, JIMMY") & (forecast_data["year"] == 1980), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "REAGAN, RONALD") & (forecast_data["year"] == 1984), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE H.W.") & (forecast_data["year"] == 1992), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CLINTON, BILL") & (forecast_data["year"] == 1996), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE W.") & (forecast_data["year"] == 2004), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "OBAMA, BARACK H.") & (forecast_data["year"] == 2012), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "TRUMP, DONALD J.") & (forecast_data["year"] == 2020), "incumbent"] = 1

# Quarter-to-quarter changes (Q2 - Q1), matching the R code
forecast_data["gdp_change"] = forecast_data["gdp_Q2"] - forecast_data["gdp_Q1"]
forecast_data["cpi_change"] = forecast_data["cpi_Q2"] - forecast_data["cpi_Q1"]
forecast_data["unemploy_change"] = forecast_data["unemployment_rate_Q2"] - forecast_data["unemployment_rate_Q1"]

# Split training (pre-2020) vs testing (2020)
forecast_data_training = forecast_data[forecast_data["year"] < 2020].copy()
forecast_data_testing  = forecast_data[forecast_data["year"] == 2020].copy()

# Fit the national OLS model
# R: vote_pct ~ incumbent * unemploy_change + party_detailed + poly(year, 2, raw = T)
# Python: use year + year^2 explicitly
train_ols = smf.ols(
    "vote_pct ~ incumbent * unemploy_change + C(party_detailed) + year + I(year**2)",
    data=forecast_data_training
).fit()

# Generate predictions for training data
forecast_data_training["pred_vote"] = train_ols.predict(forecast_data_training)
print(forecast_data_training[["vote_pct", "pred_vote"]].head(20))

# Generate predictions for test data (2020)
test_pred = train_ols.predict(forecast_data_testing)
print("\n2020 test predictions (first few):")
print(test_pred.head())


# -----------------------------------------------------------------------------
# Part 4: State-level model (poll + census + economy)
# -----------------------------------------------------------------------------
# Load pre-existing poll and census data (RDS) and convert to pandas DataFrame
# NOTE: Update the path to wherever the RDS file lives on your system.
poll_census_path = "poll_census_data.rds"
poll_census_obj = pyreadr.read_r(poll_census_path)
poll_census_data = list(poll_census_obj.values())[0]

# Prepare economic data for merging with state-level data (distinct year-level fields)
forecast_econ = forecast_data[
    ["year",
     "unemployment_rate_Q1", "unemployment_rate_Q2",
     "gdp_Q1", "gdp_Q2",
     "cpi_Q1", "cpi_Q2",
     "gdp_change", "cpi_change", "unemploy_change"]
].drop_duplicates()

# Merge state-level poll/census data with economic data
state_data = poll_census_data.merge(forecast_econ, on="year", how="left")

# Fit the state-level OLS model (training: year < 2020)
# R: vote_pct ~ poll_avg + year + party_simplified + white + black + asian + hispanic
pred_results = smf.ols(
    "vote_pct ~ poll_avg + year + C(party_simplified) + white + black + asian + hispanic",
    data=state_data[state_data["year"] < 2020]
).fit()

# Out-of-sample predictions for 2020 and beyond
out_of_sample = pred_results.predict(state_data[state_data["year"] >= 2020])

# Prepare election outcomes table (actual + predicted)
elect_outcomes = state_data[state_data["year"] >= 2020][
    ["year", "state_po", "party_simplified", "candidate", "vote_pct"]
].copy()

elect_outcomes["vote_pred"] = out_of_sample.values


# -----------------------------------------------------------------------------
# Part 5: 2020 vote difference (Biden minus Trump) and a map
# -----------------------------------------------------------------------------
# Create a 2020-only dataset
elect_2020 = elect_outcomes[elect_outcomes["year"] == 2020].copy()

# Standardize candidate names into a simple label for pivoting
elect_2020["candidate_simple"] = elect_2020["candidate"].astype(str).str.lower()
elect_2020.loc[elect_2020["candidate_simple"].str.contains("biden"), "candidate_simple"] = "biden"
elect_2020.loc[elect_2020["candidate_simple"].str.contains("trump"), "candidate_simple"] = "trump"

# Pivot wide like R pivot_wider(... names_glue = "{candidate}_{.value}")
wide_2020 = elect_2020.pivot_table(
    index=["state_po", "year"],
    columns="candidate_simple",
    values=["vote_pct", "vote_pred"],
    aggfunc="first"
)

# Flatten column names to match the R naming style (candidate_value)
wide_2020.columns = [f"{cand}_{val}" for val, cand in wide_2020.columns]
wide_2020 = wide_2020.reset_index()

# Vote difference (Biden minus Trump), matching the R intent
vote_diff_2020 = wide_2020.copy()
vote_diff_2020["vote_diff"] = vote_diff_2020["biden_vote_pct"] - vote_diff_2020["trump_vote_pct"]
vote_diff_2020 = vote_diff_2020[["state_po", "vote_diff"]].drop_duplicates()

# (Optional) Remove AK and HI to mimic the R map example
vote_diff_2020 = vote_diff_2020[~vote_diff_2020["state_po"].isin(["AK", "HI"])].copy()

# Plot a simple choropleth map of the vote difference
fig = px.choropleth(
    vote_diff_2020,
    locations="state_po",
    locationmode="USA-states",
    color="vote_diff",
    color_continuous_midpoint=0,
    scope="usa",
    title="2020 Vote Share Difference (Biden − Trump)"
)
fig.show()

# =============================================================================
# Part 3b: Clean Hold-out 2020 Evaluation
# Paste this after Part 3 in api_build_model.py
# Requires: pip install scikit-learn
#
# Models compared:
#   1. Baseline OLS (unchanged from original)
#   2. Ridge regression (L2 regularization)
#   3. Random Forest
# =============================================================================

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------------------------------------------------
# Step 1: Clean hold-out split (same logic as original, now properly evaluated)
# -----------------------------------------------------------------------------
train = forecast_data[forecast_data["year"] < 2020].copy()
test  = forecast_data[forecast_data["year"] == 2020].copy()

# -----------------------------------------------------------------------------
# Step 3: Feature set for Ridge and Random Forest
# -----------------------------------------------------------------------------
# Why Ridge over Lasso here?
#   - Very small N (~26 rows): Lasso zeroes out coefficients too aggressively
#   - Economic predictors (GDP, CPI, unemployment) are correlated with each other
#     (multicollinearity) — Ridge shrinks them evenly, Lasso arbitrarily drops some
#   - Ridge gives more stable predictions in small-sample + correlated-feature settings

forecast_data["republican"]                = (forecast_data["party_detailed"] == "REPUBLICAN").astype(int)
forecast_data["year_sq"]                   = forecast_data["year"] ** 2
forecast_data["incumbent_unemploy_change"] = forecast_data["incumbent"] * forecast_data["unemploy_change"]
forecast_data["incumbent_gdp_change"]      = forecast_data["incumbent"] * forecast_data["gdp_change"]
forecast_data["incumbent_cpi_change"]      = forecast_data["incumbent"] * forecast_data["cpi_change"]

# Rebuild train/test after adding new columns
train = forecast_data[forecast_data["year"] < 2020].copy()
test  = forecast_data[forecast_data["year"] == 2020].copy()

FEATURES = [
    "incumbent", "republican", "year", "year_sq",
    "unemployment_rate_Q1", "unemployment_rate_Q2",
    "gdp_Q1",               "gdp_Q2",
    "cpi_Q1",               "cpi_Q2",
    "unemploy_change",      "gdp_change",      "cpi_change",
    "incumbent_unemploy_change", "incumbent_gdp_change", "incumbent_cpi_change",
]

X_train = train[FEATURES].values
y_train = train["vote_pct"].values
X_test  = test[FEATURES].values

# -----------------------------------------------------------------------------
# Step 4: Ridge regression
# -----------------------------------------------------------------------------
ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),        # scale before regularization
    ("ridge",  RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                       scoring="neg_mean_squared_error"))
])
ridge_pipe.fit(X_train, y_train)
test["pred_ridge"] = ridge_pipe.predict(X_test)

# -----------------------------------------------------------------------------
# Step 5: Random Forest
# -----------------------------------------------------------------------------
# Why Random Forest?
#   - Captures non-linear relationships (e.g., incumbency effect may differ
#     across economic conditions in a non-additive way)
#   - No need to manually specify interaction terms
#   - Naturally handles multicollinearity via random feature subsampling
# Caveat: with only ~26 training rows, RF can overfit — kept shallow (max_depth=3)

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=3,          # shallow trees to avoid overfitting on small N
    min_samples_leaf=2,
    random_state=42
)
rf.fit(X_train, y_train)
test["pred_rf"] = rf.predict(X_test)

# -----------------------------------------------------------------------------
# Step 2: Baseline OLS (reproduced from original, no changes) -- MOVE IT TO HERE
# -----------------------------------------------------------------------------
baseline_ols = smf.ols(
    "vote_pct ~ incumbent * unemploy_change + C(party_detailed) + year + I(year**2)",
    data=train
).fit()
test["pred_ols"] = baseline_ols.predict(test)


# -----------------------------------------------------------------------------
# Step 6: Evaluation — MAE and RMSE for all three models
# -----------------------------------------------------------------------------
def eval_model(name, actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    print(f"  {name:<20} MAE: {mae:.4f}   RMSE: {rmse:.4f}")

print("\nHold-out 2020 — model comparison:")
eval_model("Baseline OLS",    test["vote_pct"], test["pred_ols"])
eval_model("Ridge",           test["vote_pct"], test["pred_ridge"])
eval_model("Random Forest",   test["vote_pct"], test["pred_rf"])
print(f"  (Ridge best alpha: {ridge_pipe.named_steps['ridge'].alpha_})")

print("\n2020 Predictions vs. Actuals:")
print(
    test[["candidate", "party_detailed", "vote_pct",
          "pred_ols", "pred_ridge", "pred_rf"]]
    .round(4)
    .to_string(index=False)
)

# -----------------------------------------------------------------------------
# Step 7: Plot — actual vs. predicted for all three models
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
models   = [("Baseline OLS", "pred_ols"), ("Ridge", "pred_ridge"), ("Random Forest", "pred_rf")]
colors   = {"DEMOCRAT": "#2166ac", "REPUBLICAN": "#d6604d"}

for ax, (title, col) in zip(axes, models):
    for _, row in test.iterrows():
        c = colors.get(row["party_detailed"], "gray")
        ax.scatter(row["vote_pct"], row[col], color=c, s=120,
                   edgecolors="k", linewidths=0.6, zorder=3)
        ax.annotate(row["candidate"].split(",")[0].title(),
                    (row["vote_pct"], row[col]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    lo = min(test["vote_pct"].min(), test[col].min()) - 0.02
    hi = max(test["vote_pct"].max(), test[col].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Actual vote share")
    mae = mean_absolute_error(test["vote_pct"], test[col])
    ax.text(0.05, 0.92, f"MAE={mae:.4f}", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

axes[0].set_ylabel("Predicted vote share")
plt.suptitle("Hold-out 2020: Actual vs. Predicted by Model\n(blue = Democrat, red = Republican)",
             fontsize=11)
plt.tight_layout()
plt.savefig("holdout_2020_comparison.png", dpi=150)
plt.show()




# =============================================================================
# Part 3c: Summary Table + Combined Figure for Model Comparison
# Paste after Part 3b in api_build_model.py
# =============================================================================

import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------------
# Summary table (printed + saved as CSV)
# -----------------------------------------------------------------------------
results = {
    "Model":       ["Baseline OLS", "Ridge",  "Random Forest"],
    "MAE":         [
        round(mean_absolute_error(test["vote_pct"], test["pred_ols"]),   4),
        round(mean_absolute_error(test["vote_pct"], test["pred_ridge"]), 4),
        round(mean_absolute_error(test["vote_pct"], test["pred_rf"]),    4),
    ],
    "RMSE":        [
        round(mean_squared_error(test["vote_pct"], test["pred_ols"])   ** 0.5, 4),
        round(mean_squared_error(test["vote_pct"], test["pred_ridge"]) ** 0.5, 4),
        round(mean_squared_error(test["vote_pct"], test["pred_rf"])    ** 0.5, 4),
    ],
    "Biden pred":  [
        round(test.loc[test["party_detailed"]=="DEMOCRAT",  "pred_ols"].values[0],   3),
        round(test.loc[test["party_detailed"]=="DEMOCRAT",  "pred_ridge"].values[0], 3),
        round(test.loc[test["party_detailed"]=="DEMOCRAT",  "pred_rf"].values[0],    3),
    ],
    "Trump pred":  [
        round(test.loc[test["party_detailed"]=="REPUBLICAN","pred_ols"].values[0],   3),
        round(test.loc[test["party_detailed"]=="REPUBLICAN","pred_ridge"].values[0], 3),
        round(test.loc[test["party_detailed"]=="REPUBLICAN","pred_rf"].values[0],    3),
    ],
}

summary_df = pd.DataFrame(results)
# Add actual values as reference row
actual_row = pd.DataFrame({
    "Model":      ["Actual (2020)"],
    "MAE":        ["—"],
    "RMSE":       ["—"],
    "Biden pred": [round(test.loc[test["party_detailed"]=="DEMOCRAT",  "vote_pct"].values[0], 3)],
    "Trump pred": [round(test.loc[test["party_detailed"]=="REPUBLICAN","vote_pct"].values[0], 3)],
})
summary_df = pd.concat([actual_row, summary_df], ignore_index=True)

print("\n" + "=" * 60)
print("Out-of-sample performance — Hold-out 2020")
print("=" * 60)
print(summary_df.to_string(index=False))
print("=" * 60)
summary_df.to_csv("model_comparison_table.csv", index=False)

# -----------------------------------------------------------------------------
# Combined figure: 2 panels
#   Left:  Predicted vs. Actual for all 3 models (dot = candidate)
#   Right: Residuals (actual - predicted) as horizontal bar chart
# -----------------------------------------------------------------------------
biden_actual = test.loc[test["party_detailed"]=="DEMOCRAT",  "vote_pct"].values[0]
trump_actual = test.loc[test["party_detailed"]=="REPUBLICAN","vote_pct"].values[0]

preds = {
    "Baseline OLS":  {
        "Biden": test.loc[test["party_detailed"]=="DEMOCRAT",  "pred_ols"].values[0],
        "Trump": test.loc[test["party_detailed"]=="REPUBLICAN","pred_ols"].values[0],
    },
    "Ridge": {
        "Biden": test.loc[test["party_detailed"]=="DEMOCRAT",  "pred_ridge"].values[0],
        "Trump": test.loc[test["party_detailed"]=="REPUBLICAN","pred_ridge"].values[0],
    },
    "Random Forest": {
        "Biden": test.loc[test["party_detailed"]=="DEMOCRAT",  "pred_rf"].values[0],
        "Trump": test.loc[test["party_detailed"]=="REPUBLICAN","pred_rf"].values[0],
    },
}

model_colors  = {"Baseline OLS": "#888888", "Ridge": "#4393c3", "Random Forest": "#74c476"}
cand_markers  = {"Biden": "o", "Trump": "s"}
cand_colors   = {"Biden": "#2166ac", "Trump": "#d6604d"}

fig = plt.figure(figsize=(13, 5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# --- Left panel: Predicted vs. Actual ----------------------------------------
ax1 = fig.add_subplot(gs[0])

# Reference lines for actual values
ax1.axvline(biden_actual, color="#2166ac", linewidth=1, linestyle=":", alpha=0.6, label="Biden actual")
ax1.axvline(trump_actual, color="#d6604d", linewidth=1, linestyle=":", alpha=0.6, label="Trump actual")

y_positions = {"Baseline OLS": 3, "Ridge": 2, "Random Forest": 1}
for model, cp in preds.items():
    y = y_positions[model]
    for cand, pred_val in cp.items():
        actual = biden_actual if cand == "Biden" else trump_actual
        # Line connecting actual to predicted
        ax1.plot([actual, pred_val], [y, y],
                 color=model_colors[model], linewidth=1.5, alpha=0.6)
        # Predicted dot
        ax1.scatter(pred_val, y,
                    color=cand_colors[cand], marker=cand_markers[cand],
                    s=110, zorder=5, edgecolors="k", linewidths=0.5)

ax1.set_yticks([1, 2, 3])
ax1.set_yticklabels(["Random Forest", "Ridge", "Baseline OLS"], fontsize=9)
ax1.set_xlabel("Vote share", fontsize=9)
ax1.set_title("Predicted vs. Actual — 2020 hold-out\n(circle = Biden, square = Trump)", fontsize=9)
ax1.axvline(0.5, color="black", linewidth=0.7, linestyle="--", alpha=0.4)

legend_patches = [
    mpatches.Patch(color="#2166ac", label=f"Biden (actual: {biden_actual:.3f})"),
    mpatches.Patch(color="#d6604d", label=f"Trump (actual: {trump_actual:.3f})"),
]
ax1.legend(handles=legend_patches, fontsize=8, loc="lower right")

# --- Right panel: Residuals bar chart ----------------------------------------
ax2 = fig.add_subplot(gs[1])

bar_data = []
for model, cp in preds.items():
    bar_data.append({
        "model":    model,
        "Biden":    biden_actual - cp["Biden"],
        "Trump":    trump_actual - cp["Trump"],
    })

bar_df  = pd.DataFrame(bar_data)
x       = np.arange(len(bar_df))
w       = 0.35

b1 = ax2.bar(x - w/2, bar_df["Biden"], width=w,
             color="#2166ac", alpha=0.8, label="Biden residual", edgecolor="k", linewidth=0.5)
b2 = ax2.bar(x + w/2, bar_df["Trump"], width=w,
             color="#d6604d", alpha=0.8, label="Trump residual", edgecolor="k", linewidth=0.5)

ax2.axhline(0, color="black", linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels(bar_df["model"], fontsize=9)
ax2.set_ylabel("Residual (actual − predicted)", fontsize=9)
ax2.set_title("Forecast errors by model — 2020 hold-out\n(closer to 0 = better)", fontsize=9)
ax2.legend(fontsize=8)

# Annotate bar values
for bar in b1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2,
             h + (0.01 if h >= 0 else -0.03),
             f"{h:.3f}", ha="center", va="bottom", fontsize=7)
for bar in b2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2,
             h + (0.01 if h >= 0 else -0.03),
             f"{h:.3f}", ha="center", va="bottom", fontsize=7)

plt.suptitle("Out-of-Sample Model Comparison — Hold-out 2020", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig("model_comparison_figure.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: model_comparison_figure.png  |  model_comparison_table.csv")