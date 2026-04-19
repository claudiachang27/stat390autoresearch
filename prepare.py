"""
FROZEN -- Do not modify this file.
Data loading, train/val split, evaluation metric, and plotting.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import csv
import os

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED = 42
VAL_FRACTION = 0.2
RESULTS_FILE = "results.tsv"
DATA_DIR = "data"

# ── Data ───────────────────────────────────────────────────
def load_data():
    """
    Load and merge Crunchbase CSVs.
    Target: did the company reach a Series A round? (1 = yes, 0 = no)
    """
    # Load CSVs
    companies    = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-companies.csv"), encoding="latin-1")
    rounds       = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-rounds.csv"), encoding="latin-1")
    investments  = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-investments.csv"), encoding="latin-1", low_memory=False)
    acquisitions = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-acquisitions.csv"), encoding="latin-1")

    # ── Target: reached Series A ───────────────────────────
    series_a_companies = set(
        rounds.loc[
            rounds["funding_round_type"].str.lower().str.strip() == "series-a",
            "company_permalink"
        ].dropna()
    )
    companies["reached_series_a"] = companies["permalink"].isin(series_a_companies).astype(int)

    # ── Feature: total seed funding ────────────────────────
    seed_rounds = rounds[rounds["funding_round_type"].str.lower().str.strip() == "seed"]
    seed_total = (
        seed_rounds.groupby("company_permalink")["raised_amount_usd"]
        .sum()
        .reset_index()
        .rename(columns={"raised_amount_usd": "total_seed_usd"})
    )
    companies = companies.merge(seed_total, left_on="permalink", right_on="company_permalink", how="left")
    companies.drop(columns=["company_permalink"], inplace=True)

    # ── Feature: number of seed rounds ─────────────────────
    seed_count = (
        seed_rounds.groupby("company_permalink")
        .size()
        .reset_index(name="num_seed_rounds")
    )
    companies = companies.merge(seed_count, left_on="permalink", right_on="company_permalink", how="left")
    companies.drop(columns=["company_permalink"], inplace=True)

    # ── Feature: number of total investors ─────────────────
    investor_count = (
        investments.groupby("company_permalink")
        .size()
        .reset_index(name="num_investors")
    )
    companies = companies.merge(investor_count, left_on="permalink", right_on="company_permalink", how="left")
    companies.drop(columns=["company_permalink"], inplace=True)

    # ── Feature: number of distinct investors ──────────────
    distinct_investors = (
        investments.groupby("company_permalink")["investor_permalink"]
        .nunique()
        .reset_index(name="num_distinct_investors")
    )
    companies = companies.merge(distinct_investors, left_on="permalink", right_on="company_permalink", how="left")
    companies.drop(columns=["company_permalink"], inplace=True)

    # ── Feature: was company acquired ──────────────────────
    acquired = set(acquisitions["company_permalink"].dropna())
    companies["was_acquired"] = companies["permalink"].isin(acquired).astype(int)

    # ── Feature: days from founding to first funding ───────
    companies["founded_at"]       = pd.to_datetime(companies["founded_at"],       errors="coerce")
    companies["first_funding_at"] = pd.to_datetime(companies["first_funding_at"], errors="coerce")
    companies["days_to_first_funding"] = (
        companies["first_funding_at"] - companies["founded_at"]
    ).dt.days

    # ── Feature: categorical encodings ─────────────────────
    companies["category_code_enc"] = companies["category_code"].astype("category").cat.codes
    companies["country_code_enc"]  = companies["country_code"].astype("category").cat.codes
    companies["state_code_enc"]    = companies["state_code"].astype("category").cat.codes

    # ── Assemble feature matrix ─────────────────────────────
    FEATURE_COLS = [
        "funding_total_usd",
        "funding_rounds",
        "total_seed_usd",
        "num_seed_rounds",
        "num_investors",
        "num_distinct_investors",
        "was_acquired",
        "days_to_first_funding",
        "category_code_enc",
        "country_code_enc",
        "state_code_enc",
    ]

    X = companies[FEATURE_COLS].copy().fillna(-1).values.astype(float)
    y = companies["reached_series_a"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_FRACTION, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, y_train, X_val, y_val, FEATURE_COLS


# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, X_val, y_val):
    """Compute validation AUC-ROC (higher is better)."""
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1]
    else:
        y_score = model.decision_function(X_val)
    return float(roc_auc_score(y_val, y_score))


# ── Logging ────────────────────────────────────────────────
def log_result(experiment_id, val_auc, status, description):
    """Append one row to results.tsv."""
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow(["experiment", "val_auc", "status", "description"])
        writer.writerow([experiment_id, f"{val_auc:.6f}", status, description])


# ── Plotting ───────────────────────────────────────────────
def plot_results(save_path="performance.png"):
    """Plot validation AUC-ROC over experiments from results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    experiments, aucs, statuses, descriptions = [], [], [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiments.append(r
