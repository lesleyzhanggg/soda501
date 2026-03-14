###############################################################################
# LLM-Assisted Data Extraction (Human-in-the-Loop)
# FREE / LOCAL VERSION (Hugging Face)
# Structured Extraction from Messy Text Using LOCAL LLMs (Python)
# Author: Jared Edgerton
# Date: date.today()
#
# Teaching version:
# - Explicit, sequential steps
# - Avoids high-level pipeline() wrapper
#
# Install (recommend pinning for class):
#   pip install -U "transformers>=4.40.0" "torch" "sentencepiece" "pydantic>=2.0" pandas numpy scikit-learn
###############################################################################

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
import os

os.environ["HF_TOKEN"] = "hf_"
torch
import json
import re

import numpy as np
import pandas as pd

import torch

from datetime import date
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

np.random.seed(123)
set_seed(123)

# ---------------------------------------------------------------------
# Part 1: Define a Schema (What fields do we want?)
# ---------------------------------------------------------------------
EventType = Literal[
    "protest",
    "election",
    "policy_change",
    "violence",
    "disaster",
    "other"
]

GeoPrecision = Literal[
    "country_only",
    "admin1_or_state",
    "city_or_local",
    "unknown"
]

class EvidenceSpan(BaseModel):
    field: Literal["event_type", "date", "location", "actors", "outcome"]
    quote: str

class EventExtraction(BaseModel):
    doc_id: str

    # NOTE (teaching/demo setting):
    # For local small models, strict JSON/schema enforcement often yields empty outputs.
    # We therefore provide safe defaults so the pipeline can produce "messy but usable"
    # records while still flagging uncertainty. In a production setting, tighten these
    # requirements and fail fast.

    event_type: EventType = "other"

    event_date_iso: Optional[str] = Field(
        default=None,
        description="ISO date YYYY-MM-DD if available; otherwise null."
    )
    date_is_approximate: bool = Field(
        default=True,
        description="True if the date is estimated/inferred (e.g., 'early April')."
    )

    country: Optional[str] = None
    admin1_or_state: Optional[str] = None
    city_or_local: Optional[str] = None
    geo_precision: GeoPrecision = "unknown"

    actors: List[str] = Field(default_factory=list, description="Key actors mentioned (individuals, orgs, groups).")

    outcome_summary: Optional[str] = Field(
        default=None,
        description="One-sentence outcome summary (what happened)."
    )

    extraction_confidence: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="Model self-rated confidence (0 to 1)."
    )
    uncertainty_flags: List[str] = Field(
        default_factory=list,
        description="List of issues that make extraction uncertain (e.g., missing date, vague location)."
    )
    evidence: List[EvidenceSpan] = Field(
        default_factory=list,
        description="Short quotes supporting each extracted field (if available)."
    )


# ---------------------------------------------------------------------
# Part 2: Create Messy Text Inputs (Mini Corpus)
# ---------------------------------------------------------------------
docs = [
    {"doc_id": "doc_001", "text": "Breaking: Thousands rallied in Santiago on 2026-03-14 demanding pension reform. Police reported minor clashes; 12 were arrested."},
    {"doc_id": "doc_002", "text": "On March 2nd, lawmakers passed the 'Clean Air Act' amendment in the national assembly. Environmental groups praised the vote."},
    {"doc_id": "doc_003", "text": "Election officials said voting will take place next Sunday. Turnout is expected to be high in the capital."},
    {"doc_id": "doc_004", "text": "A 6.2 magnitude earthquake struck near the coastal city overnight, damaging dozens of homes and cutting power to 40,000 residents."},
    {"doc_id": "doc_005", "text": "Witnesses described gunfire outside a nightclub late Friday; at least two people were injured, but details remain unclear."},
    {"doc_id": "doc_006", "text": "The governor announced a new curfew order effective immediately. Critics called it an overreach."},
    {"doc_id": "doc_007", "text": "Early April saw renewed demonstrations in the northern province after fuel prices rose again."},
    {"doc_id": "doc_008", "text": "Floodwaters inundated low-lying neighborhoods; emergency shelters opened at local schools, officials said."},
    {"doc_id": "doc_009", "text": "Opposition leaders met with international observers in Brussels to discuss election monitoring."},
    {"doc_id": "doc_010", "text": "Police said the suspect was arrested after a stabbing in downtown; the mayor urged calm."},
    {"doc_id": "doc_011", "text": "Parliament reversed the prior ban on rideshare apps, citing labor market flexibility."},
    {"doc_id": "doc_012", "text": "A protest was planned for tomorrow, but organizers postponed it due to severe weather warnings."},
    {"doc_id": "doc_013", "text": "Following a landslide, the ministry declared a state of emergency in two districts."},
    {"doc_id": "doc_014", "text": "The court ruling sparked demonstrations across the city center; human rights groups condemned the decision."},
    {"doc_id": "doc_015", "text": "The article mentions reforms and elections in passing but gives no clear time or place."},
]

docs_df = pd.DataFrame(docs)

print("\n------------------------------")
print("Input corpus (first 5 docs)")
print("------------------------------")
print(docs_df.head())
print("docs_df shape:", docs_df.shape)

# ---------------------------------------------------------------------
# Part 3: Prompt Design (Schemas + Guardrails)
# ---------------------------------------------------------------------
json_template = {
    "doc_id": "doc_XXX",
    "event_type": "other",
    "event_date_iso": None,
    "date_is_approximate": False,
    "country": None,
    "admin1_or_state": None,
    "city_or_local": None,
    "geo_precision": "unknown",
    "actors": [],
    "outcome_summary": None,
    "extraction_confidence": 0.5,
    "uncertainty_flags": [],
    "evidence": [
        {"field": "event_type", "quote": ""},
        {"field": "date", "quote": ""},
        {"field": "location", "quote": ""},
        {"field": "actors", "quote": ""},
        {"field": "outcome", "quote": ""}
    ]
}

system_instructions = (
    "Task: Extract ONE event record from the text.\n"
    "Return EXACTLY the following 9 lines, one per line, in the format key: value\n"
    "Use empty value if unknown.\n"
    "\n"
    "event_type: protest|election|policy_change|violence|disaster|other\n"
    "event_date_iso: YYYY-MM-DD\n"
    "date_is_approximate: true|false\n"
    "country:\n"
    "admin1_or_state:\n"
    "city_or_local:\n"
    "geo_precision: country_only|admin1_or_state|city_or_local|unknown\n"
    "actors: comma-separated list\n"
    "outcome_summary: one sentence\n"
    "\n"
    "Do not output anything else.\n"
)

# ---------------------------------------------------------------------
# Part 4: Local LLM Structured Extraction (Batch Processing)
# ---------------------------------------------------------------------
# Model choice: small, free, runs on CPU (slow but fine for class demos)
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("\n------------------------------")
print("Loading tokenizer + model")
print("------------------------------")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

use_gpu = torch.cuda.is_available()
device = torch.device("cuda") if use_gpu else torch.device("cpu")
model = model.to(device)

print("Model:", model_name)
print("Device:", device)

extractions = []

print("\n------------------------------")
print("Running LOCAL LLM extraction (one doc at a time)")
print("------------------------------")

for i in range(len(docs_df)):
    doc_id = docs_df.loc[i, "doc_id"]
    text = docs_df.loc[i, "text"]

    prompt = (
        f"{system_instructions}\n"
        f"Document ID: {doc_id}\n"
        f"Text: {text}\n"
    )

    # 1) Tokenize (explicit)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 2) Generate (explicit)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )

    # 3) Decode (explicit)
    out_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # 4) Parse labeled key:value lines (best-effort)
    parse_ok = True
    parse_error = ""
    parse_flags: List[str] = []

    allowed_keys = {
        "event_type",
        "event_date_iso",
        "date_is_approximate",
        "country",
        "admin1_or_state",
        "city_or_local",
        "geo_precision",
        "actors",
        "outcome_summary"
    }

    lines = [ln.strip() for ln in out_text.splitlines() if ":" in ln]
    kv = {}

    for ln in lines:
        k, v = ln.split(":", 1)
        k_norm = k.strip().lower()
        if k_norm in allowed_keys:
            kv[k_norm] = v.strip()

    if len(kv) == 0:
        parse_ok = False
        parse_error = "no_key_value_lines_found"
        parse_flags.append("parse_failed_local_model_output")

    event_type = (kv.get("event_type", "other") or "other").strip().lower()
    if event_type not in {"protest", "election", "policy_change", "violence", "disaster", "other"}:
        parse_flags.append("invalid_event_type_from_model")
        event_type = "other"

    event_date_iso = (kv.get("event_date_iso", "") or "").strip() or None

    date_is_approx_raw = (kv.get("date_is_approximate", "") or "").strip().lower()
    if date_is_approx_raw in {"true", "false"}:
        date_is_approximate = (date_is_approx_raw == "true")
    else:
        parse_flags.append("date_is_approximate_missing_or_invalid")
        date_is_approximate = True

    country = (kv.get("country", "") or "").strip() or None
    admin1_or_state = (kv.get("admin1_or_state", "") or "").strip() or None
    city_or_local = (kv.get("city_or_local", "") or "").strip() or None

    geo_precision = (kv.get("geo_precision", "unknown") or "unknown").strip().lower()
    if geo_precision not in {"country_only", "admin1_or_state", "city_or_local", "unknown"}:
        parse_flags.append("invalid_geo_precision_from_model")
        geo_precision = "unknown"

    actors_raw = (kv.get("actors", "") or "").strip()
    actors = [a.strip() for a in actors_raw.split(",") if a.strip()] if actors_raw else []

    outcome_summary = (kv.get("outcome_summary", "") or "").strip() or None

    extracted_obj = EventExtraction(
        doc_id=doc_id,
        event_type=event_type,
        event_date_iso=event_date_iso,
        date_is_approximate=date_is_approximate,
        country=country,
        admin1_or_state=admin1_or_state,
        city_or_local=city_or_local,
        geo_precision=geo_precision,
        actors=actors,
        outcome_summary=outcome_summary,
        extraction_confidence=0.35 if parse_ok else 0.2,
        uncertainty_flags=parse_flags,
        evidence=[]
    )

    extra_dict = extracted_obj.model_dump()


    # 5) Attach trace fields (explicit)
    extra_dict["raw_text"] = text
    extra_dict["local_model_raw_output"] = out_text
    extra_dict["parse_ok"] = parse_ok
    extra_dict["parse_error"] = parse_error

    # 6) Flatten list fields for CSV (explicit)
    extra_dict["evidence_json"] = json.dumps(extra_dict["evidence"], ensure_ascii=False)
    extra_dict["uncertainty_flags_json"] = json.dumps(extra_dict["uncertainty_flags"], ensure_ascii=False)
    extra_dict.pop("evidence")
    extra_dict.pop("uncertainty_flags")

    extractions.append(extra_dict)

# 7) Build dataframe + save
extractions_df = pd.DataFrame(extractions)

print("\n------------------------------")
print("Extracted records (first 5 rows)")
print("------------------------------")
print(extractions_df.head())
print("extractions_df shape:", extractions_df.shape)

os.makedirs("outputs", exist_ok=True)
extractions_df.to_csv("outputs/extractions_raw.csv", index=False)

# ---------------------------------------------------------------------
# Part 5: Uncertainty Checks (Automatic Flags for Human Review)
# ---------------------------------------------------------------------
extractions_df["extraction_confidence"] = pd.to_numeric(extractions_df["extraction_confidence"], errors="coerce")

extractions_df["flag_parse_failed"] = ~extractions_df["parse_ok"]
extractions_df["flag_low_confidence"] = extractions_df["extraction_confidence"] < 0.70
extractions_df["flag_missing_date"] = extractions_df["event_date_iso"].isna()
extractions_df["flag_missing_country"] = extractions_df["country"].isna()
extractions_df["flag_geo_unknown"] = extractions_df["geo_precision"].isin(["unknown", "country_only"])

flag_cols = [
    "flag_parse_failed",
    "flag_low_confidence",
    "flag_missing_date",
    "flag_missing_country",
    "flag_geo_unknown"
]
extractions_df["needs_human_review"] = extractions_df[flag_cols].any(axis=1)

print("\n------------------------------")
print("Review flag counts")
print("------------------------------")
print(extractions_df[flag_cols + ["needs_human_review"]].sum(numeric_only=True))

extractions_df.to_csv("outputs/extractions_with_flags.csv", index=False)

# ---------------------------------------------------------------------
# Part 6: Human Validation / Spot-Audits (Create an Audit Sheet)
# ---------------------------------------------------------------------
audit_random_n = 5
audit_random = extractions_df.sample(n=audit_random_n, random_state=123)

audit_flagged = extractions_df[extractions_df["needs_human_review"]].copy()

audit_sheet = pd.concat([audit_random, audit_flagged], ignore_index=True).drop_duplicates(subset=["doc_id"])
audit_sheet = audit_sheet.sort_values("doc_id").reset_index(drop=True)

audit_sheet["human_is_correct"] = ""
audit_sheet["human_correct_event_type"] = ""
audit_sheet["human_correct_date_iso"] = ""
audit_sheet["human_correct_location"] = ""
audit_sheet["failure_mode"] = ""
audit_sheet["reviewer_notes"] = ""

audit_sheet.to_csv("outputs/human_audit_sheet.csv", index=False)

print("\n------------------------------")
print("Wrote outputs/human_audit_sheet.csv")
print("------------------------------")

# ---------------------------------------------------------------------
# Part 7: Evaluation Patterns (Precision/Recall + Auditing)
# ---------------------------------------------------------------------
gold = pd.DataFrame([
    {"doc_id": "doc_001", "event_type_gold": "protest"},
    {"doc_id": "doc_002", "event_type_gold": "policy_change"},
    {"doc_id": "doc_003", "event_type_gold": "election"},
    {"doc_id": "doc_004", "event_type_gold": "disaster"},
    {"doc_id": "doc_005", "event_type_gold": "violence"},
    {"doc_id": "doc_006", "event_type_gold": "policy_change"},
    {"doc_id": "doc_007", "event_type_gold": "protest"},
    {"doc_id": "doc_008", "event_type_gold": "disaster"},
])

eval_df = gold.merge(extractions_df[["doc_id", "event_type"]], on="doc_id", how="left")
eval_df = eval_df.rename(columns={"event_type": "event_type_pred"})
eval_df["event_type_pred"] = eval_df["event_type_pred"].fillna("MISSING")

print("\n------------------------------")
print("Evaluation table (gold vs predicted)")
print("------------------------------")
print(eval_df)

print("\n------------------------------")
print("Classification report (event_type)")
print("------------------------------")
print(classification_report(eval_df["event_type_gold"], eval_df["event_type_pred"], zero_division=0))




###############################################################################
# Q6: Evaluation + Prompt Iteration
# Paste this AFTER the full llm_human.py pipeline
# (i.e., after Part 7, so model, tokenizer, docs_df, gold are all defined)
#
# This script:
#   1. Defines two new prompt variants
#   2. Re-runs extraction for both
#   3. Computes classification report for each
#   4. Produces a comparison table (macro-F1, accuracy, review queue size)
#   5. Saves all outputs to outputs/
###############################################################################

from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

# =============================================================================
# Helper: run extraction with a given prompt template and return a dataframe
# =============================================================================
def run_extraction(prompt_fn, docs_df, model, tokenizer, device, label):
    """
    prompt_fn: function(doc_id, text) -> prompt string
    label:     string name for this prompt variant
    """
    records = []
    print(f"\n--- Running extraction: {label} ---")

    for i in range(len(docs_df)):
        doc_id = docs_df.loc[i, "doc_id"]
        text   = docs_df.loc[i, "text"]
        prompt = prompt_fn(doc_id, text)

        inputs       = tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids    = inputs["input_ids"].to(device)
        attn_mask    = inputs["attention_mask"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        out_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # --- Parse key: value lines ---
        allowed_keys = {
            "event_type", "event_date_iso", "date_is_approximate",
            "country", "admin1_or_state", "city_or_local",
            "geo_precision", "actors", "outcome_summary"
        }
        lines = [ln.strip() for ln in out_text.splitlines() if ":" in ln]
        kv = {}
        for ln in lines:
            k, v = ln.split(":", 1)
            k_norm = k.strip().lower()
            if k_norm in allowed_keys:
                kv[k_norm] = v.strip()

        parse_ok    = len(kv) > 0
        parse_flags = [] if parse_ok else ["parse_failed"]

        event_type = (kv.get("event_type", "other") or "other").strip().lower()
        if event_type not in {"protest", "election", "policy_change", "violence", "disaster", "other"}:
            parse_flags.append("invalid_event_type")
            event_type = "other"

        country        = (kv.get("country", "") or "").strip() or None
        event_date_iso = (kv.get("event_date_iso", "") or "").strip() or None
        geo_precision  = (kv.get("geo_precision", "unknown") or "unknown").strip().lower()
        if geo_precision not in {"country_only", "admin1_or_state", "city_or_local", "unknown"}:
            geo_precision = "unknown"
        actors_raw = (kv.get("actors", "") or "").strip()
        actors     = [a.strip() for a in actors_raw.split(",") if a.strip()]

        confidence = 0.35 if parse_ok else 0.2

        # --- Review flags (same 5 as original) ---
        flag_parse_failed    = not parse_ok
        flag_low_confidence  = confidence < 0.70
        flag_missing_date    = event_date_iso is None
        flag_missing_country = country is None
        flag_geo_unknown     = geo_precision in ["unknown", "country_only"]
        needs_review = any([
            flag_parse_failed, flag_low_confidence,
            flag_missing_date, flag_missing_country, flag_geo_unknown
        ])

        records.append({
            "doc_id":             doc_id,
            "event_type":         event_type,
            "event_date_iso":     event_date_iso,
            "country":            country,
            "geo_precision":      geo_precision,
            "actors":             actors,
            "extraction_confidence": confidence,
            "parse_ok":           parse_ok,
            "uncertainty_flags":  parse_flags,
            "needs_human_review": needs_review,
            "raw_output":         out_text,
            "prompt_variant":     label,
        })
        print(f"  {doc_id}: event_type={event_type}  parse_ok={parse_ok}")

    return pd.DataFrame(records)


# =============================================================================
# Define 3 prompt variants
# =============================================================================

# --- Baseline prompt (same as original llm_human.py) ---
baseline_instructions = (
    "Task: Extract ONE event record from the text.\n"
    "Return EXACTLY the following 9 lines, one per line, in the format key: value\n"
    "Use empty value if unknown.\n"
    "\n"
    "event_type: protest|election|policy_change|violence|disaster|other\n"
    "event_date_iso: YYYY-MM-DD\n"
    "date_is_approximate: true|false\n"
    "country:\n"
    "admin1_or_state:\n"
    "city_or_local:\n"
    "geo_precision: country_only|admin1_or_state|city_or_local|unknown\n"
    "actors: comma-separated list\n"
    "outcome_summary: one sentence\n"
    "\n"
    "Do not output anything else.\n"
)

def prompt_baseline(doc_id, text):
    return f"{baseline_instructions}\nDocument ID: {doc_id}\nText: {text}\n"


# --- Variant 1: explicit missingness rules + stronger evidence requirement ---
# Change: (a) tells model to write "null" explicitly when unknown (not leave blank)
#         (b) requires a one-word evidence quote for event_type
variant1_instructions = (
    "Task: Extract ONE event record from the text.\n"
    "Return EXACTLY the following 10 lines, one per line, in the format key: value\n"
    "If a field is unknown or not mentioned, write null as the value.\n"
    "Do NOT leave any field blank.\n"
    "\n"
    "event_type: protest|election|policy_change|violence|disaster|other\n"
    "event_type_evidence: paste the exact word or phrase from the text that justifies event_type\n"
    "event_date_iso: YYYY-MM-DD or null\n"
    "date_is_approximate: true|false\n"
    "country: country name or null\n"
    "admin1_or_state: state/province or null\n"
    "city_or_local: city or null\n"
    "geo_precision: country_only|admin1_or_state|city_or_local|unknown\n"
    "actors: comma-separated list or null\n"
    "outcome_summary: one sentence summarizing what happened, or null\n"
    "\n"
    "Do not output anything else.\n"
)

def prompt_variant1(doc_id, text):
    return f"{variant1_instructions}\nDocument ID: {doc_id}\nText: {text}\n"


# --- Variant 2: shorter taxonomy with definitions to reduce event_type confusion ---
# Change: adds brief definition for each event_type category
variant2_instructions = (
    "Task: Extract ONE event record from the text.\n"
    "Return EXACTLY the following 9 lines, one per line, in the format key: value\n"
    "Use null if a field is unknown.\n"
    "\n"
    "Event type definitions:\n"
    "  protest    = demonstration, rally, strike, march\n"
    "  election   = voting, ballot, electoral process\n"
    "  policy_change = law passed, regulation changed, government decision\n"
    "  violence   = shooting, stabbing, armed conflict, assault\n"
    "  disaster   = earthquake, flood, fire, natural or industrial emergency\n"
    "  other      = does not clearly fit the above\n"
    "\n"
    "event_type: protest|election|policy_change|violence|disaster|other\n"
    "event_date_iso: YYYY-MM-DD or null\n"
    "date_is_approximate: true|false\n"
    "country: or null\n"
    "admin1_or_state: or null\n"
    "city_or_local: or null\n"
    "geo_precision: country_only|admin1_or_state|city_or_local|unknown\n"
    "actors: comma-separated list or null\n"
    "outcome_summary: one sentence or null\n"
    "\n"
    "Do not output anything else.\n"
)

def prompt_variant2(doc_id, text):
    return f"{variant2_instructions}\nDocument ID: {doc_id}\nText: {text}\n"


# =============================================================================
# Run all three variants
# =============================================================================
df_baseline = run_extraction(prompt_baseline,  docs_df, model, tokenizer, device, "baseline")
df_v1       = run_extraction(prompt_variant1,  docs_df, model, tokenizer, device, "variant1_evidence")
df_v2       = run_extraction(prompt_variant2,  docs_df, model, tokenizer, device, "variant2_definitions")

# Save raw extraction CSVs
df_baseline.to_csv("outputs/q6_extractions_baseline.csv",  index=False)
df_v1.to_csv(      "outputs/q6_extractions_variant1.csv",  index=False)
df_v2.to_csv(      "outputs/q6_extractions_variant2.csv",  index=False)


# =============================================================================
# Evaluate each variant against the gold set
# =============================================================================
def evaluate(df_extracted, gold, label):
    eval_df = gold.merge(
        df_extracted[["doc_id", "event_type", "needs_human_review"]],
        on="doc_id", how="left"
    ).rename(columns={"event_type": "event_type_pred"})
    eval_df["event_type_pred"] = eval_df["event_type_pred"].fillna("MISSING")

    y_true = eval_df["event_type_gold"]
    y_pred = eval_df["event_type_pred"]

    report     = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    macro_f1   = report["macro avg"]["f1-score"]
    accuracy   = accuracy_score(y_true, y_pred)
    review_n   = df_extracted["needs_human_review"].sum()
    review_pct = review_n / len(df_extracted) * 100

    print(f"\n{'='*50}")
    print(f"Variant: {label}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "prompt_variant":   label,
        "macro_F1":         round(macro_f1,  3),
        "accuracy":         round(accuracy,  3),
        "review_queue_n":   int(review_n),
        "review_queue_pct": round(review_pct, 1),
    }

results = [
    evaluate(df_baseline, gold, "baseline"),
    evaluate(df_v1,       gold, "variant1_evidence"),
    evaluate(df_v2,       gold, "variant2_definitions"),
]

# =============================================================================
# Comparison table
# =============================================================================
comparison_df = pd.DataFrame(results)
print("\n" + "="*60)
print("PROMPT COMPARISON TABLE")
print("="*60)
print(comparison_df.to_string(index=False))

comparison_df.to_csv("outputs/q6_prompt_comparison.csv", index=False)
print("\nSaved: outputs/q6_prompt_comparison.csv")
print("Saved: outputs/q6_extractions_baseline/variant1/variant2.csv")