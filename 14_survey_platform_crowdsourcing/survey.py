###############################################################################
# Survey Exports + Cleaning + Codebooks: Python (HK Selective Exposure Data)
# Adapted from Jared Edgerton's tutorial
# Dataset: 2018 Selective Exposure HK News Survey
#
# This script covers Questions 4-9 of the problem set.
###############################################################################

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
import os
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("data_raw", exist_ok=True)
os.makedirs("data_processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =============================================================================
# Question 4: Survey export — load + inspect
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Load the .dta file
# -----------------------------------------------------------------------------
survey_raw = pd.read_stata("2018Selective_Exposure_HK_news_merged_20190430.dta")

# -----------------------------------------------------------------------------
# Step 2: Print and inspect
# -----------------------------------------------------------------------------
print("\n==============================")
print("Q4: Dataset shape")
print("==============================")
print("Rows:", survey_raw.shape[0], " Columns:", survey_raw.shape[1])

print("\n==============================")
print("Q4: Variable types (first 30)")
print("==============================")
print(survey_raw.dtypes.head(30))

print("\n==============================")
print("Q4: First 5 rows (key variables)")
print("==============================")
print(survey_raw[["v1", "v6", "v8", "v9", "uid", "consent",
                   "age", "sex", "edu", "q1", "q2", "q3"]].head())

# Five platform-export variables:
# 1. v1       = ResponseId (unique survey response identifier)
# 2. v8       = StartDate (survey start timestamp)
# 3. v9       = EndDate (survey end timestamp)
# 4. v6       = IPAddress (respondent IP)
# 5. uid      = Participant ID (platform-assigned unique ID)
# Also: consent, time_pre (duration in seconds), satisfice (attention flag)

print("\n==============================")
print("Q4: Potential cleaning issue")
print("==============================")
print("time_pre has extreme outliers (max =", survey_raw["time_pre"].max(),
      "seconds = ~5 days), likely from respondents leaving the browser open.")
print("These need to be capped or flagged before using duration as a quality check.")

# =============================================================================
# Question 5: Cleaning — names, types, and missing values
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Standardize column names to snake_case
# -----------------------------------------------------------------------------
clean_cols = [
    re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", c)).strip("_").lower()
    for c in survey_raw.columns
]
survey_clean = survey_raw.copy()
survey_clean.columns = clean_cols

# Rename platform metadata to meaningful names
survey_clean = survey_clean.rename(columns={
    "v1": "response_id",
    "v2": "response_set",
    "v3": "name",
    "v6": "ip_address",
    "v7": "v7_flag",
    "v8": "start_date",
    "v9": "end_date",
    "v10": "finished",
})

print("\n==============================")
print("Q5: Cleaned column names (first 20)")
print("==============================")
print(list(survey_clean.columns[:20]))

# -----------------------------------------------------------------------------
# Step 2: Parse start/end timestamps into datetime
# -----------------------------------------------------------------------------
survey_clean["start_date"] = pd.to_datetime(
    survey_clean["start_date"], format="%Y/%m/%d %H:%M", errors="coerce"
)
survey_clean["end_date"] = pd.to_datetime(
    survey_clean["end_date"], format="%Y/%m/%d %H:%M", errors="coerce"
)

# -----------------------------------------------------------------------------
# Step 3: Convert variables to appropriate types
# -----------------------------------------------------------------------------
# age is already numeric (float); ensure it stays that way
survey_clean["age"] = pd.to_numeric(survey_clean["age"], errors="coerce")

# duration: time_pre is in seconds, convert to minutes
survey_clean["duration_min"] = survey_clean["time_pre"] / 60.0

# -----------------------------------------------------------------------------
# Step 4: Recode missing-value strings / codes into NaN
# -----------------------------------------------------------------------------
# consent: 1 = Yes, 2 = No; NaN = missing
# Many survey items use special codes (e.g., 8 or 98) for "don't know" / refusal
survey_items = ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
for col in survey_items:
    survey_clean[col] = survey_clean[col].replace({8: np.nan, 9: np.nan, 98: np.nan, 99: np.nan})

# sex: recode NaN values
survey_clean["sex"] = survey_clean["sex"].replace({99: np.nan})

print("\n==============================")
print("Q5: Key variables after cleaning")
print("==============================")
print(survey_clean[["response_id", "start_date", "end_date", "time_pre",
                      "duration_min", "consent", "age", "sex"]].head(10))

# =============================================================================
# Question 6: Codebook — document your data
# =============================================================================
codebook = pd.DataFrame({
    "variable": [
        "response_id", "start_date", "end_date", "time_pre",
        "duration_min", "ip_address", "uid", "consent",
        "age", "sex", "edu", "polint",
        "ideology", "q1", "q2", "q3",
        "satisfice", "totalclick", "condition", "selective",
    ],
    "description": [
        "Unique response identifier (Qualtrics)",
        "Survey start timestamp (parsed datetime)",
        "Survey end timestamp (parsed datetime)",
        "Survey duration in seconds (platform metadata)",
        "Survey duration in minutes (derived from time_pre)",
        "Respondent IP address (sensitive metadata)",
        "Platform-assigned unique participant ID",
        "Consent to participate (1=Yes, 2=No)",
        "Age in years (numeric)",
        "Sex (0=female, 1=male)",
        "Education level (0=low, 0.5=mid, 1=high)",
        "Political interest (numeric scale)",
        "Political ideology (numeric scale)",
        "Survey item Q1 (numeric, 8/9 recoded to NA)",
        "Survey item Q2 (numeric, 8/9 recoded to NA)",
        "Survey item Q3 (numeric, 8/9 recoded to NA)",
        "Satisficing flag from original data (0=no, 1=mild, 2=severe)",
        "Total article clicks in experiment",
        "Experimental condition assignment",
        "Selective exposure measure",
    ],
    "notes": [
        "Qualtrics-generated, format R_xxxx",
        "Parsed from v8; format YYYY/MM/DD HH:MM",
        "Parsed from v9; format YYYY/MM/DD HH:MM",
        "Extreme outliers present (max ~454k sec); cap or flag",
        "Derived: time_pre / 60",
        "Treat as sensitive; do not publish",
        "Hashed panel ID",
        "Drop if consent != 1",
        "Range 18-70 in this sample",
        "Binary coded",
        "3-level ordinal",
        "Higher = more interested",
        "Higher = more conservative",
        "8 and 9 recoded to NA (don't know / refused)",
        "Same recoding as Q1",
        "Same recoding as Q1",
        "Pre-computed by original researchers",
        "Count of articles clicked during experiment",
        "Experimental group assignment",
        "Computed selective exposure score",
    ],
})

print("\n==============================")
print("Q6: Codebook (20 variables)")
print("==============================")
print(codebook.to_string(index=False))

codebook.to_csv("outputs/week_codebook.csv", index=False)
print("\nSaved: outputs/week_codebook.csv")

# =============================================================================
# Question 7: Labeling — variable labels + value labels
# =============================================================================

# Variable labels (stored as metadata dictionary)
var_labels = {
    "response_id": "Unique response identifier",
    "time_pre": "Survey duration (seconds)",
    "age": "Age in years",
    "sex": "Sex (binary)",
    "consent": "Consent to participate",
    "ideology": "Political ideology",
    "polint": "Political interest",
}
survey_clean.attrs["var_labels"] = var_labels

# Value labels: create a labeled numeric variable for consent
consent_labels = {1: "Yes", 2: "No"}
survey_clean.attrs["value_labels_consent"] = consent_labels

# Value labels: create a labeled numeric variable for sex
sex_labels = {0: "Female", 1: "Male"}
survey_clean.attrs["value_labels_sex"] = sex_labels

# Create a labeled education variable
edu_labels = {0.0: "Low", 0.5: "Mid", 1.0: "High"}
survey_clean.attrs["value_labels_edu"] = edu_labels

# Print excerpt showing labels
print("\n==============================")
print("Q7: Variable labels")
print("==============================")
print(survey_clean.attrs["var_labels"])

print("\n==============================")
print("Q7: Value labels")
print("==============================")
print("Consent:", survey_clean.attrs["value_labels_consent"])
print("Sex:", survey_clean.attrs["value_labels_sex"])
print("Education:", survey_clean.attrs["value_labels_edu"])

print("\n==============================")
print("Q7: Labeled variables (sample)")
print("==============================")
labeled_sample = survey_clean[["response_id", "consent", "sex", "edu", "age"]].head(10).copy()
labeled_sample["consent_label"] = labeled_sample["consent"].map(consent_labels)
labeled_sample["sex_label"] = labeled_sample["sex"].map(sex_labels)
labeled_sample["edu_label"] = labeled_sample["edu"].map(edu_labels)
print(labeled_sample)

# =============================================================================
# Question 8: Quality checks — flags + summary
# =============================================================================

# -----------------------------------------------------------------------------
# Flag 1: No consent
# -----------------------------------------------------------------------------
survey_clean["flag_no_consent"] = survey_clean["consent"] != 1

# -----------------------------------------------------------------------------
# Flag 2: Speeding (duration < 120 seconds)
# -----------------------------------------------------------------------------
survey_clean["flag_fast"] = survey_clean["time_pre"] < 120

# -----------------------------------------------------------------------------
# Flag 3: Attention check / satisficing
# -----------------------------------------------------------------------------
# The dataset includes a pre-computed satisfice variable (0=none, 1=mild, 2=severe)
# Flag respondents with severe satisficing (== 2)
survey_clean["flag_attention_fail"] = survey_clean["satisfice"] >= 2

# -----------------------------------------------------------------------------
# Flag 4: Missingness — share missing across key survey items
# -----------------------------------------------------------------------------
key_vars = survey_clean[["q1", "q2", "q3", "q4", "q5", "q6", "q7",
                          "age", "sex", "edu", "polint", "ideology"]]
survey_clean["missing_share"] = key_vars.isna().mean(axis=1)
survey_clean["flag_missing_high"] = survey_clean["missing_share"] > 0.30

# -----------------------------------------------------------------------------
# Flag 5: Straightlining — identical responses across q1-q7
# -----------------------------------------------------------------------------
q_cols = ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
q_data = survey_clean[q_cols]
# Standard deviation of 0 across all items (ignoring NaN) = straightlining
q_std = q_data.std(axis=1, skipna=True)
survey_clean["flag_straightline"] = (q_std == 0) & (q_data.notna().sum(axis=1) >= 4)

# -----------------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------------
flag_summary = pd.DataFrame({
    "n_total": [len(survey_clean)],
    "n_no_consent": [survey_clean["flag_no_consent"].sum()],
    "n_fast": [survey_clean["flag_fast"].sum()],
    "n_attention_fail": [survey_clean["flag_attention_fail"].sum()],
    "n_missing_high": [survey_clean["flag_missing_high"].sum()],
    "n_straightline": [survey_clean["flag_straightline"].sum()],
})

print("\n==============================")
print("Q8: Flag summary")
print("==============================")
print(flag_summary.to_string(index=False))

# Full flag table for all respondents
flagged_view = survey_clean[[
    "response_id", "time_pre", "consent", "satisfice", "missing_share",
    "flag_no_consent", "flag_fast", "flag_attention_fail",
    "flag_missing_high", "flag_straightline",
]].sort_values(
    by=["flag_no_consent", "flag_fast", "flag_attention_fail"],
    ascending=[False, False, False]
)

print("\n==============================")
print("Q8: Flag table (first 15 rows)")
print("==============================")
print(flagged_view.head(15).to_string(index=False))

# =============================================================================
# Question 9: Analysis-ready dataset — filter + save + visualize
# =============================================================================

# -----------------------------------------------------------------------------
# Filter out flagged respondents
# -----------------------------------------------------------------------------
print("\n==============================")
print("Q9: Row counts before/after filtering")
print("==============================")
print("Before:", len(survey_clean))

survey_final = survey_clean.loc[
    (~survey_clean["flag_no_consent"]) &
    (~survey_clean["flag_fast"]) &
    (~survey_clean["flag_attention_fail"]) &
    (~survey_clean["flag_missing_high"])
].copy()

print("After: ", len(survey_final))

# Save cleaned dataset
survey_final.to_csv("data_processed/week_survey_clean.csv", index=False)
print("Saved: data_processed/week_survey_clean.csv")

# -----------------------------------------------------------------------------
# Diagnostic plot 1: Duration histogram
# -----------------------------------------------------------------------------
# Cap at 3000 seconds for visualization (outliers compress the histogram)
duration_capped = survey_clean["time_pre"].clip(upper=3000)

plt.figure(figsize=(8, 4))
plt.hist(duration_capped.dropna(), bins=50, edgecolor="white", alpha=0.8)
plt.axvline(x=120, color="red", linestyle="--", linewidth=1, label="Speeding threshold (120s)")
plt.title("Survey duration distribution (capped at 3000s)")
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("figures/duration_histogram.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# Diagnostic plot 2: Missingness histogram
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.hist(survey_clean["missing_share"].dropna(), bins=20, edgecolor="white", alpha=0.8)
plt.axvline(x=0.30, color="red", linestyle="--", linewidth=1, label="High-missingness threshold (30%)")
plt.title("Missingness share across key variables")
plt.xlabel("Share missing (per respondent)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("figures/missingness_histogram.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# Diagnostic plot 3: Flags summary bar chart
# -----------------------------------------------------------------------------
flag_names = ["No consent", "Speeding", "Attention fail", "High missing", "Straightlining"]
flag_counts = [
    survey_clean["flag_no_consent"].sum(),
    survey_clean["flag_fast"].sum(),
    survey_clean["flag_attention_fail"].sum(),
    survey_clean["flag_missing_high"].sum(),
    survey_clean["flag_straightline"].sum(),
]

plt.figure(figsize=(8, 4))
plt.bar(flag_names, flag_counts, color=["#534AB7", "#1D9E75", "#D85A30", "#378ADD", "#D4537E"])
plt.title("Respondents flagged by each quality check")
plt.xlabel("Flag type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/flags_summary.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSaved all figures to figures/")
print("Done.")
