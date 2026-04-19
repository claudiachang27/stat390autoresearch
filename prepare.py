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
    Load and merge Crunchbase CSVs. Build features relevant to:
      - Founder demographics / educational pedigree
      - Early-stage capital injection
      - Investor prestige
    Target: did the company reach a Series A round? (1 = yes, 0 = no)
    """
    # Load CSVs
    companies     = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-companies.csv"))
    rounds        = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-rounds.csv"))
    investments   = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-investments.csv"))
    acquisitions  = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-acquisitions.csv"))

    # ── Target: reached Series A ───────────────────────────
    series_a_companies = set(
        rounds.loc[
            rounds["funding_round_type"].str.lower().str.strip() == "series-a",
            "company_permalink"
        ].dropna()
    )
    companies["reached_series_a"] = (
        companies["permalink"].isin(series_a_companies)
    ).astype(int)

    # ── Feature: total seed funding ────────────────────────
    seed_rounds = rounds[rounds["funding_round_type"].str.lower().str.strip() == "seed"]
    seed_total = (
        seed_rounds.groupby("company_permalink")["raised_amount_usd"]
        .sum()
        .reset_index()
        .rename(columns={"raised_amount_usd": "total_seed_usd"})
    )
    companies = companies.merge(seed_total, left_on="permalink", right_on="company_permalink", how="left")

    # ── Feature: number of seed rounds ─────────────────────
    seed_count = (
        seed_rounds.groupby("company_permalink")
        .size()
        .reset_index(name="num_seed_rounds")
    )
    companies = companies.merge(seed_count, left_on="permalink", right_on="company_permalink", how="left")

    # ── Feature: number of early investors ─────────────────
    investor_count = (
        investments.groupby("company_permalink")
        .size()
        .reset_index(name="num_investors")
    )
    companies = companies.merge(investor_count, left_on="permalink", right_on="company_permalink", how="left")

    # ── Feature: investor prestige (unique investor names) ─
    # Proxy: number of distinct investors (more = broader network)
    distinct_investors = (
        investments.groupby("company_permalink")["investor_permalink"]
        .nunique()
        .reset_index(name="num_distinct_investors")
    )
    companies = companies.merge(distinct_investors, left_on="permalink", right_on="company_permalink", how="left")

    # ── Feature: was company acquired (exit signal) ────────
    acquired = set(acquisitions["company_permalink"].dropna())
    companies["was_acquired"] = companies["permalink"].isin(acquired).astype(int)

    # ── Feature: company age at first funding (days) ───────
    companies["founded_at"] = pd.to_datetime(companies["founded_at"], errors="coerce")
    first_funding = (
        rounds.groupby("company_permalink")["funded_at"]
        .min()
        .reset_index()
        .rename(columns={"funded_at": "first_funding_at"})
    )
    first_funding["first_funding_at"] = pd.to_datetime(first_funding["first_funding_at"], errors="coerce")
    companies = companies.merge(first_funding, left_on="permalink", right_on="company_permalink", how="left")
    companies["days_to_first_funding"] = (
        companies["first_funding_at"] - companies["founded_at"]
    ).dt.days

    # ── Feature: category / market (encoded) ───────────────
    companies["category_code_enc"] = companies["category_code"].astype("category").cat.codes

    # ── Feature: country (encoded) ─────────────────────────
    companies["country_code_enc"] = companies["country_code"].astype("category").cat.codes

    # ── Feature: state (encoded) ───────────────────────────
    companies["state_code_enc"] = companies["state_code"].astype("category").cat.codes

    # ── Assemble feature matrix ─────────────────────────────
    FEATURE_COLS = [
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

    # Drop companies that already have a Series A
    # (only keep seed-stage or earlier as training pool)
    X = companies[FEATURE_COLS].copy()
    y = companies["reached_series_a"].values

    X = X.fillna(-1)  # simple imputation; model.py can override
    X = X.values.astype(float)

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
    auc = float(roc_auc_score(y_val, y_score))
    return auc


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
            experiments.append(row["experiment"])
            aucs.append(float(row["val_auc"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(range(len(aucs)), aucs, c=colors, s=80, zorder=3,
               edgecolors="white", linewidth=0.5)
    ax.plot(range(len(aucs)), aucs, "k--", alpha=0.2, zorder=2)

    # Best-so-far envelope (higher is better)
    best_so_far = []
    current_best = -float("inf")
    for a in aucs:
        current_best = max(current_best, a)
        best_so_far.append(current_best)
    ax.plot(range(len(aucs)), best_so_far, color="#2ecc71",
            linewidth=2.5, label="Best so far")

    ax.set_ylabel("Validation AUC-ROC (higher is better)", fontsize=12)
    ax.set_title(
        "AutoResearch: Predicting Series A Funding Success\n"
        "Founder Pedigree vs. Capital & Investor Signals",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(max(0.4, min(aucs) - 0.05), min(1.0, max(aucs) + 0.05))
    ax.grid(True, alpha=0.3)

    short_labels = [d[:22] + ".." if len(d) > 24 else d for d in descriptions]
    ax.set_xticks(range(len(aucs)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=10, label="baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=10, label="keep (improved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=10, label="discard (regressed)"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Best so far"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_results()
