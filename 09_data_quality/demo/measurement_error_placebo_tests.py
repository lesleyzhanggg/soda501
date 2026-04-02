###############################################################################
# Measurement Error + Placebo Tests Tutorial: Python
# Author: Jared Edgerton
# Revised: bugs in Q5 problem-set section fixed
#
# This script demonstrates:
#   1) Classic measurement error in covariates (X_true vs X_observed)
#   2) How measurement error can bias regression estimates (including confounding)
#   3) Simple correction idea using a "validation subsample" (regression calibration)
#   4) Placebo tests as pipeline diagnostics:
#        - Outcome placebo (negative control outcome)
#        - Treatment permutation placebo (randomization inference)
#
# Dependencies: numpy, pandas, matplotlib (no statsmodels required)
#
# Teaching note:
# - Written as a sequential workflow so students can see how the pipeline unfolds.
# - No user-defined functions (no def ...).
# - Minimal control flow (loops are used for simulation/permutations).
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(123)

# Create common project folders (safe to run repeatedly)
os.makedirs("data_raw",       exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("figures",        exist_ok=True)
os.makedirs("outputs",        exist_ok=True)
os.makedirs("src",            exist_ok=True)

# -----------------------------------------------------------------------------
# Helper: OLS via numpy (no statsmodels needed)
# We use numpy.linalg.lstsq to fit OLS: y = X @ beta
# This avoids requiring statsmodels or scipy.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Part 1: A simple data generating process with confounding
# -----------------------------------------------------------------------------
# We simulate:
#   X_true: a true confounder (unobserved truth)
#   D: a treatment/exposure correlated with X_true
#   Y: an outcome affected by both D and X_true
#
# Then we observe:
#   X_obs = X_true + U   (measurement error in X)
#
# Key intuition:
# - If X_true is a confounder, we need to control for it to estimate the effect of D.
# - If we only have a noisy measure X_obs, we do not fully control for confounding.
# - As measurement error increases, bias in the estimated effect of D can increase.

n = 5000

# True confounder
x_true = np.random.normal(loc=0.0, scale=1.0, size=n)

# Treatment assignment correlated with x_true (logistic link)
# (This creates confounding: D is not independent of x_true.)
logit_p = 1.0 * x_true
p = 1.0 / (1.0 + np.exp(-logit_p))
d = np.random.binomial(n=1, p=p, size=n)

# True outcome model
tau  = 1.0    # true effect of D on Y
beta = 1.0    # effect of X_true on Y
eps_y = np.random.normal(loc=0.0, scale=1.0, size=n)

y = tau * d + beta * x_true + eps_y

# Placebo outcome (negative control outcome): NOT affected by D by construction
eps_pl    = np.random.normal(loc=0.0, scale=1.0, size=n)
y_placebo = 0.0 * d + beta * x_true + eps_pl

df_base = pd.DataFrame({"y": y, "y_placebo": y_placebo, "d": d, "x_true": x_true})

print("\n--- Data preview ---")
print(df_base.head())
print("\nTreatment rate:", round(df_base["d"].mean(), 4))

# -----------------------------------------------------------------------------
# Part 2: Measurement error settings
# -----------------------------------------------------------------------------
# We will vary the measurement error in X:
#   X_obs = X_true + U,  U ~ Normal(0, sigma_u)
#
# For each sigma_u, we will estimate:
#   (A) Oracle model: y ~ d + x_true           (best case; uses truth)
#   (B) Naive model:  y ~ d + x_obs            (what you do with a noisy measure)
#   (C) Calibration:  estimate x_true ~ x_obs in a validation sample, predict x_hat,
#                     then run y ~ d + x_hat
#
# We will also run an outcome placebo:
#   y_placebo ~ d + x_obs
#
# NOTE: This is synthetic, so we *know* x_true. In real work, "x_true" might be
# available only in a validation dataset or via higher-quality measurement.

sigma_u_grid = [0.0, 0.2, 0.5, 1.0, 2.0]
R = 30  # repetitions (to see variability from measurement error draws)

# Fix a "validation sample" index set (20% of observations)
validation_share = 0.20
val_size         = int(validation_share * n)
validation_idx   = np.random.choice(np.arange(n), size=val_size, replace=False)
is_validation    = np.zeros(n, dtype=bool)
is_validation[validation_idx] = True

# Storage
rows = []

print("\n--- Running measurement error simulations (tutorial) ---")
for sigma_u in sigma_u_grid:
    tau_oracle_list  = []
    tau_naive_list   = []
    tau_cal_list     = []
    tau_placebo_list = []

    beta_oracle_list = []
    beta_naive_list  = []
    beta_cal_list    = []

    for r in range(R):
        # Draw measurement error and observed covariate
        u     = np.random.normal(loc=0.0, scale=sigma_u, size=n)
        x_obs = x_true + u

        # Build design matrices with intercept column
        ones = np.ones(n)

        # (A) Oracle regression: y ~ 1 + d + x_true
        X_oracle              = np.column_stack([ones, d, x_true])
        coef_oracle, _, _, _  = np.linalg.lstsq(X_oracle, y, rcond=None)
        tau_oracle_list.append(coef_oracle[1])
        beta_oracle_list.append(coef_oracle[2])

        # (B) Naive regression: y ~ 1 + d + x_obs
        X_naive              = np.column_stack([ones, d, x_obs])
        coef_naive, _, _, _  = np.linalg.lstsq(X_naive, y, rcond=None)
        tau_naive_list.append(coef_naive[1])
        beta_naive_list.append(coef_naive[2])

        # (C) Regression calibration (validation subsample):
        #     estimate x_true ~ x_obs on validation sample, predict x_hat for all.
        ones_val             = np.ones(val_size)                           # matches is_validation
        X_cal_val            = np.column_stack([ones_val, x_obs[is_validation]])
        coef_cal, _, _, _    = np.linalg.lstsq(X_cal_val, x_true[is_validation], rcond=None)
        x_hat                = coef_cal[0] + coef_cal[1] * x_obs

        X_calibrated             = np.column_stack([ones, d, x_hat])
        coef_calibrated, _, _, _ = np.linalg.lstsq(X_calibrated, y, rcond=None)
        tau_cal_list.append(coef_calibrated[1])
        beta_cal_list.append(coef_calibrated[2])

        # Outcome placebo: y_placebo ~ 1 + d + x_obs
        X_placebo              = np.column_stack([ones, d, x_obs])
        coef_placebo, _, _, _  = np.linalg.lstsq(X_placebo, y_placebo, rcond=None)
        tau_placebo_list.append(coef_placebo[1])

    # Summaries per sigma_u
    rows.append(
        {
            "sigma_u":          sigma_u,
            "tau_true":         tau,
            "tau_oracle_mean":  float(np.mean(tau_oracle_list)),
            "tau_naive_mean":   float(np.mean(tau_naive_list)),
            "tau_cal_mean":     float(np.mean(tau_cal_list)),
            "tau_placebo_mean": float(np.mean(tau_placebo_list)),
            "tau_oracle_q025":  float(np.quantile(tau_oracle_list,  0.025)),
            "tau_oracle_q975":  float(np.quantile(tau_oracle_list,  0.975)),
            "tau_naive_q025":   float(np.quantile(tau_naive_list,   0.025)),
            "tau_naive_q975":   float(np.quantile(tau_naive_list,   0.975)),
            "tau_cal_q025":     float(np.quantile(tau_cal_list,     0.025)),
            "tau_cal_q975":     float(np.quantile(tau_cal_list,     0.975)),
            "beta_true":        beta,
            "beta_oracle_mean": float(np.mean(beta_oracle_list)),
            "beta_naive_mean":  float(np.mean(beta_naive_list)),
            "beta_cal_mean":    float(np.mean(beta_cal_list)),
        }
    )
    print(f"  done sigma_u={sigma_u}")

results = pd.DataFrame(rows)
print("\n--- Summary (means) ---")
print(results[["sigma_u", "tau_true", "tau_oracle_mean", "tau_naive_mean",
               "tau_cal_mean", "tau_placebo_mean"]].to_string(index=False))

results.to_csv("outputs/measurement_error_results.csv", index=False)

# -----------------------------------------------------------------------------
# Part 3: Plot how measurement error changes estimates
# -----------------------------------------------------------------------------

# Plot tau estimates vs sigma_u
plt.figure(figsize=(8, 5))
plt.plot(results["sigma_u"], results["tau_oracle_mean"],  marker="o", label="Oracle: y ~ d + x_true")
plt.plot(results["sigma_u"], results["tau_naive_mean"],   marker="o", label="Naive: y ~ d + x_obs")
plt.plot(results["sigma_u"], results["tau_cal_mean"],     marker="o", label="Calibration: y ~ d + x_hat")
plt.plot(results["sigma_u"], results["tau_placebo_mean"], marker="o", label="Outcome placebo: y_pl ~ d + x_obs")
plt.axhline(tau, linestyle="--", label="True tau")
plt.title("Estimated treatment effect vs measurement error in confounder")
plt.xlabel("Measurement error SD (sigma_u)")
plt.ylabel("Estimated coefficient on d")
plt.legend()
plt.tight_layout()
plt.savefig("figures/measurement_error_tau_vs_sigma.png", dpi=200)
plt.close()

# Plot beta estimates vs sigma_u (confounder coefficient attenuation)
plt.figure(figsize=(8, 5))
plt.plot(results["sigma_u"], results["beta_oracle_mean"], marker="o", label="Oracle: coef on x_true")
plt.plot(results["sigma_u"], results["beta_naive_mean"],  marker="o", label="Naive: coef on x_obs")
plt.plot(results["sigma_u"], results["beta_cal_mean"],    marker="o", label="Calibration: coef on x_hat")
plt.axhline(beta, linestyle="--", label="True beta")
plt.title("Estimated confounder effect vs measurement error (attenuation)")
plt.xlabel("Measurement error SD (sigma_u)")
plt.ylabel("Estimated coefficient on confounder term")
plt.legend()
plt.tight_layout()
plt.savefig("figures/measurement_error_beta_vs_sigma.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# Part 4: Treatment permutation placebo (randomization inference)
# -----------------------------------------------------------------------------
# Here we treat the observed estimate as a test statistic:
#   tau_hat_obs = coef on d in y ~ d + x_obs
# Then we build a null distribution by permuting d.

sigma_u_perm = 1.0
u_perm       = np.random.normal(loc=0.0, scale=sigma_u_perm, size=n)
x_obs_perm   = x_true + u_perm

ones   = np.ones(n)
X_obs  = np.column_stack([ones, d, x_obs_perm])
coef_obs, _, _, _ = np.linalg.lstsq(X_obs, y, rcond=None)
tau_hat_obs = float(coef_obs[1])

print("\n--- Permutation placebo setup ---")
print("sigma_u used:", sigma_u_perm)
print("Observed tau_hat (naive model):", round(tau_hat_obs, 4))

B        = 500
tau_perm = []

for b in range(B):
    d_perm           = np.random.permutation(d)
    X_b              = np.column_stack([ones, d_perm, x_obs_perm])
    coef_b, _, _, _  = np.linalg.lstsq(X_b, y, rcond=None)
    tau_perm.append(float(coef_b[1]))

tau_perm = np.array(tau_perm)

# Empirical two-sided p-value
p_emp = (1.0 + np.sum(np.abs(tau_perm) >= np.abs(tau_hat_obs))) / (B + 1.0)
print("Empirical p-value (two-sided):", round(p_emp, 4))

perm_df = pd.DataFrame({"tau_perm": tau_perm})
perm_df.to_csv("outputs/permutation_tau_distribution.csv", index=False)

# Plot permutation distribution + observed line
plt.figure(figsize=(8, 5))
plt.hist(tau_perm, bins=30, alpha=0.8)
plt.axvline( tau_hat_obs, linestyle="--", linewidth=2,
             label=f"Observed tau_hat = {tau_hat_obs:.3f}")
plt.axvline(-tau_hat_obs, linestyle="--", linewidth=1)
plt.title(f"Treatment permutation placebo (sigma_u={sigma_u_perm})\n"
          f"Empirical p-value = {p_emp:.3f}")
plt.xlabel("Coefficient on permuted treatment")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("figures/permutation_placebo_tau_hist.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# End of tutorial
# -----------------------------------------------------------------------------
print("\nDone. Outputs written to:")
print("  outputs/measurement_error_results.csv")
print("  outputs/permutation_tau_distribution.csv")
print("  figures/measurement_error_tau_vs_sigma.png")
print("  figures/measurement_error_beta_vs_sigma.png")
print("  figures/permutation_placebo_tau_hist.png")

###############################################################################
# PROBLEM SET
###############################################################################

# =============================================================================
# Q5. Validation subsample and regression calibration.
#
# Fix sigma_u = 1.0. Loop over validation_share in [0.05, 0.20, 0.50].
# For each share: draw a correctly-sized validation index set, run R repetitions,
# and report mean calibrated vs naive tau.
#
# Bug fixes vs. original stub
# ----------------------------
# BUG 1 (TypeError):  validation_share was defined as a list [0.05, 0.2, 0.5]
#   and then immediately used as a scalar in int(validation_share * n).
#   Fix: rename the list to validation_shares; use loop variable vs as the scalar.
#
# BUG 2 (ValueError – shape mismatch): ones_val was built outside the share loop
#   using the stale scalar, so its length didn't match x_obs[is_validation] when
#   is_validation was rebuilt for a different share.
#   Fix: compute val_size = int(vs * n) at the top of the outer loop and derive
#   ones_val = np.ones(val_size) inside the inner loop, after val_size is known.
# =============================================================================

sigma_u_q5        = 1.0
R                 = 30
validation_shares = [0.05, 0.20, 0.50]   # list declared here; never used as scalar

q5_rows = []

print("\n--- Q5: Validation subsample × regression calibration ---")
print(f"    Fixed sigma_u = {sigma_u_q5}")

for vs in validation_shares:             # vs is always a plain float
    val_size = int(vs * n)               # scalar: 250 / 1000 / 2500

    # Draw a correctly-sized validation index set for this share
    validation_idx = np.random.choice(np.arange(n), size=val_size, replace=False)
    is_validation  = np.zeros(n, dtype=bool)
    is_validation[validation_idx] = True

    tau_naive_list = []
    tau_cal_list   = []

    for r in range(R):
        u     = np.random.normal(loc=0.0, scale=sigma_u_q5, size=n)
        x_obs = x_true + u
        ones  = np.ones(n)

        # (B) Naive regression: y ~ 1 + d + x_obs
        X_naive              = np.column_stack([ones, d, x_obs])
        coef_naive, _, _, _  = np.linalg.lstsq(X_naive, y, rcond=None)
        tau_naive_list.append(float(coef_naive[1]))

        # (C) Regression calibration (validation subsample)
        ones_val          = np.ones(val_size)                    # always matches is_validation
        X_cal_val         = np.column_stack([ones_val, x_obs[is_validation]])
        coef_cal, _, _, _ = np.linalg.lstsq(
            X_cal_val, x_true[is_validation], rcond=None
        )
        x_hat = coef_cal[0] + coef_cal[1] * x_obs

        X_calibrated             = np.column_stack([ones, d, x_hat])
        coef_calibrated, _, _, _ = np.linalg.lstsq(X_calibrated, y, rcond=None)
        tau_cal_list.append(float(coef_calibrated[1]))

    tau_naive_mean = float(np.mean(tau_naive_list))
    tau_cal_mean   = float(np.mean(tau_cal_list))

    q5_rows.append({
        "validation_share": vs,
        "val_n":            val_size,
        "sigma_u":          sigma_u_q5,
        "tau_true":         tau,
        "tau_naive_mean":   tau_naive_mean,
        "tau_cal_mean":     tau_cal_mean,
        "bias_naive":       tau_naive_mean - tau,
        "bias_cal":         tau_cal_mean   - tau,
    })

    print(f"\n  validation_share = {vs:.2f}  (n_val = {val_size})")
    print(f"    tau_naive_mean = {tau_naive_mean:.4f}   (bias = {tau_naive_mean - tau:+.4f})")
    print(f"    tau_cal_mean   = {tau_cal_mean:.4f}   (bias = {tau_cal_mean   - tau:+.4f})")

q5_df = pd.DataFrame(q5_rows)
q5_df.to_csv("outputs/q5_validation_calibration_results.csv", index=False)

# Q5 plot: naive vs calibrated tau by validation share
fig, ax = plt.subplots(figsize=(8, 5))
xp = np.arange(len(validation_shares))
w  = 0.28

naive_means = [r["tau_naive_mean"] for r in q5_rows]
cal_means   = [r["tau_cal_mean"]   for r in q5_rows]

ax.bar(xp - w/2, naive_means, width=w, alpha=0.8, label="Naive: y ~ d + x_obs")
ax.bar(xp + w/2, cal_means,   width=w, alpha=0.8, label="Calibrated: y ~ d + x_hat")
ax.axhline(tau, linestyle="--", color="black", label=f"True tau = {tau}")
ax.set_xticks(xp)
ax.set_xticklabels([f"{v:.0%}" for v in validation_shares])
ax.set_xlabel("Validation share")
ax.set_ylabel("Mean estimated tau  (R=30 repetitions)")
ax.set_title(f"Q5 — Naive vs Calibrated Treatment Estimate\n"
             f"(sigma_u = {sigma_u_q5})")
ax.legend()
plt.tight_layout()
plt.savefig("figures/q5_calibration_by_validation_share.png", dpi=200)
plt.close()
print("\n  Saved: figures/q5_calibration_by_validation_share.png")

# =============================================================================
# Q6. Placebo tests: outcome placebo and treatment permutation.
# =============================================================================

print("\n--- Q6: Placebo tests  (sigma_u = 1.0) ---")

sigma_u_q6 = 1.0
u_q6       = np.random.normal(loc=0.0, scale=sigma_u_q6, size=n)
x_obs_q6   = x_true + u_q6
ones       = np.ones(n)

# ── 6a. Outcome placebo ──────────────────────────────────────────────────────
# y_placebo was generated with true coefficient 0 on d.
# Regressing y_placebo ~ d + x_obs should yield a coefficient on d near zero;
# any deviation reflects residual confounding from imperfect control via x_obs.

X_pl                = np.column_stack([ones, d, x_obs_q6])
coef_pl, _, _, _    = np.linalg.lstsq(X_pl, y_placebo, rcond=None)
tau_placebo_obs     = float(coef_pl[1])

print("\n  6a. Outcome placebo")
print(f"      Coef on d in y_placebo ~ d + x_obs:  {tau_placebo_obs:.4f}")
print("      (True coefficient = 0 by construction)")

# ── 6b. Treatment permutation placebo ────────────────────────────────────────
# Observed naive estimate
X_obs_q6              = np.column_stack([ones, d, x_obs_q6])
coef_obs_q6, _, _, _  = np.linalg.lstsq(X_obs_q6, y, rcond=None)
tau_hat_obs_q6        = float(coef_obs_q6[1])

print("\n  6b. Treatment permutation placebo")
print(f"      Observed naive tau_hat:  {tau_hat_obs_q6:.4f}")

B_q6      = 500
tau_perm_q6 = []

for b in range(B_q6):
    d_perm           = np.random.permutation(d)
    X_b              = np.column_stack([ones, d_perm, x_obs_q6])
    coef_b, _, _, _  = np.linalg.lstsq(X_b, y, rcond=None)
    tau_perm_q6.append(float(coef_b[1]))

tau_perm_q6 = np.array(tau_perm_q6)
p_emp_q6    = (1.0 + np.sum(np.abs(tau_perm_q6) >= np.abs(tau_hat_obs_q6))) / (B_q6 + 1.0)

print(f"      Permutation dist. mean:  {tau_perm_q6.mean():.4f}")
print(f"      Permutation dist. SD:    {tau_perm_q6.std():.4f}")
print(f"      Empirical p-value (two-sided):  {p_emp_q6:.4f}")

pd.DataFrame({"tau_perm": tau_perm_q6}).to_csv(
    "outputs/q6_permutation_tau_distribution.csv", index=False
)

# Q6 permutation histogram
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(tau_perm_q6, bins=35, alpha=0.80, color="steelblue",
        edgecolor="white", linewidth=0.4, label="Permutation null distribution")
ax.axvline( tau_hat_obs_q6, linestyle="--", linewidth=2.0, color="crimson",
            label=f"Observed tau_hat = {tau_hat_obs_q6:.3f}")
ax.axvline(-tau_hat_obs_q6, linestyle=":",  linewidth=1.5, color="crimson",
            label=f"Mirror = {-tau_hat_obs_q6:.3f}")
ax.set_xlabel("Coefficient on (permuted) treatment d")
ax.set_ylabel(f"Count  (B = {B_q6} permutations)")
ax.set_title(f"Q6 — Treatment Permutation Placebo  (sigma_u = {sigma_u_q6})\n"
             f"Observed tau_hat = {tau_hat_obs_q6:.3f}    "
             f"Empirical p-value = {p_emp_q6:.4f}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/q6_permutation_placebo_tau_hist.png", dpi=200)
plt.close()
print("\n  Saved: figures/q6_permutation_placebo_tau_hist.png")

# =============================================================================
# Final summary
# =============================================================================
print("\n══════════════════════════════════════════════")
print("PROBLEM SET SUMMARY")
print("══════════════════════════════════════════════")
print("\nQ5 — Calibration results (sigma_u = 1.0, R = 30):")
print(q5_df[["validation_share", "val_n", "tau_naive_mean",
             "tau_cal_mean", "bias_naive", "bias_cal"]].to_string(index=False))
print(f"\nQ6a — Outcome placebo coef on d:     {tau_placebo_obs:.4f}  (true = 0)")
print(f"Q6b — Observed naive tau_hat:        {tau_hat_obs_q6:.4f}")
print(f"Q6b — Empirical p-value (2-sided):   {p_emp_q6:.4f}")
print("\nAll outputs written to outputs/ and figures/")