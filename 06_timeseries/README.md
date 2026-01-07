# Week 14

**Theme:** Time Series Analyses for Social Data

## Goals
- Recognize common time-indexed social data structures (time series vs TSCS panels vs event streams) and why the structure determines modeling + validation. :contentReference[oaicite:0]{index=0}
- Avoid the biggest evaluation mistake in time-indexed data: **time leakage** (random splits that let the model “see the future”). Use **train on past, test on future** as the default. :contentReference[oaicite:1]{index=1}
- Implement a minimal forecasting workflow in R and evaluate honestly using a time split and rolling-origin backtesting. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
- Diagnose dependence over time (trend, seasonality, autocorrelation) using decomposition and ACF/PACF, and connect patterns to plausible DGPs. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
- Introduce “causal timing” designs when interventions occur at a known time (Interrupted Time Series), including placebo checks. :contentReference[oaicite:6]{index=6}

## Materials
- `demo/` — in-class demo code (if present); main script: `timeseries.R` :contentReference[oaicite:7]{index=7}
- Slides: `timeseries.pdf` :contentReference[oaicite:8]{index=8}
- Use `/data/raw`, `/data/intermediate`, `/data/processed` for datasets and outputs
- Environment/setup notes live in `/environment`
- Recommended outputs structure (used in exercises):
  - `outputs/figures/` (e.g., decomposition, backtest, ITS plots)
  - `outputs/tables/` (e.g., backtest errors, ITS results) :contentReference[oaicite:9]{index=9}

## Notes
- **Default rule:** evaluation must mimic deployment (train on past → test on future). Use rolling windows/backtesting when possible. :contentReference[oaicite:10]{index=10}
- Time-series work is sensitive to time zones, aggregation rules, missingness handling, and feature-window choices—log these for reproducibility. :contentReference[oaicite:11]{index=11}

## Readings
- Chen et al. (2022) PA
- Park and Yamauchi (2022) PA
