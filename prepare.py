"""
FROZEN -- Do not modify this file.
Data loading, train/val split, evaluation metric, and plotting.
"""
"""
prepare.py  --  Data loading + optional feature engineering.

Mirrors the frozen helper.py interface:
    X_train, y_train, X_val, y_val, feature_cols = load_data()

Additionally exposes:
    X_train_eng, X_val_eng, eng_cols = engineer_features(X_train, X_val, feature_cols)

Workflow this file supports
────────────────────────────
  Step 1 – Baseline (LogisticRegression on raw features)
           → use load_data() only

  Step 2 – Feature engineering
           → call engineer_features() after load_data()
           → re-run LogisticRegression to see whether engineered
             features close the gap with non-linear models

  Step 3 – Non-linear models (XGBoost, RandomForest)
           → feed them the engineered feature matrix from Step 2

  Step 4 – Hyperparameter tuning on whichever model won Step 3
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import os

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED = 42
VAL_FRACTION = 0.20
DATA_DIR = "data"

# Raw numeric features fed to every model.
# Categorical encodings are kept last so slicing [:NUM_NUMERIC] is easy.
FEATURE_COLS = [
    "funding_total_usd",
    "funding_rounds",
    "total_seed_usd",
    "num_seed_rounds",
    "num_investors",
    "num_distinct_investors",
    "was_acquired",
    "days_to_first_funding",
    # ── new raw features ──────────────────────────────────
    "num_other_rounds",        # angel + convertible_note + … rounds (non-seed, non-A)
    "avg_seed_round_size",     # total_seed_usd / num_seed_rounds  (0 when no seed)
    "investor_per_round",      # num_investors / funding_rounds    (density signal)
    "has_us_investor",         # 1 if any investor is US-based
    # ── categorical encodings (kept at the end) ───────────
    "category_code_enc",
    "country_code_enc",
    "state_code_enc",
]

# The categorical columns at the tail should NOT get polynomial expansion.
# Everything before this index is treated as numeric for engineer_features().
_N_NUMERIC = FEATURE_COLS.index("category_code_enc")   # 12


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Raw data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """
    Load and merge Crunchbase CSVs, build raw features, return train/val split.

    Returns
    -------
    X_train, y_train, X_val, y_val  :  np.ndarray
    feature_cols                     :  list[str]  (same order as columns in X)
    """
    companies    = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-companies.csv"),    encoding="latin-1")
    rounds       = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-rounds.csv"),       encoding="latin-1")
    investments  = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-investments.csv"),  encoding="latin-1", low_memory=False)
    acquisitions = pd.read_csv(os.path.join(DATA_DIR, "crunchbase-acquisitions.csv"), encoding="latin-1")

    # ── Target ─────────────────────────────────────────────
    series_a_permalinks = set(
        rounds.loc[
            rounds["funding_round_type"].str.lower().str.strip() == "series-a",
            "company_permalink",
        ].dropna()
    )
    companies["reached_series_a"] = companies["permalink"].isin(series_a_permalinks).astype(int)

    # ── Seed aggregates ────────────────────────────────────
    seed_rounds = rounds[rounds["funding_round_type"].str.lower().str.strip() == "seed"]

    seed_total = (
        seed_rounds.groupby("company_permalink")["raised_amount_usd"]
        .sum()
        .reset_index(name="total_seed_usd")
    )
    seed_count = (
        seed_rounds.groupby("company_permalink")
        .size()
        .reset_index(name="num_seed_rounds")
    )

    # ── Other (non-seed, non-series-a) rounds ─────────────
    other_rounds = rounds[
        ~rounds["funding_round_type"].str.lower().str.strip().isin(["seed", "series-a"])
    ]
    other_count = (
        other_rounds.groupby("company_permalink")
        .size()
        .reset_index(name="num_other_rounds")
    )

    # ── Investor aggregates ────────────────────────────────
    investor_count = (
        investments.groupby("company_permalink")
        .size()
        .reset_index(name="num_investors")
    )
    distinct_investors = (
        investments.groupby("company_permalink")["investor_permalink"]
        .nunique()
        .reset_index(name="num_distinct_investors")
    )

    # Has at least one US-based investor?
    us_investors = (
        investments[investments["investor_country_code"].str.upper().str.strip() == "USA"]
        .groupby("company_permalink")
        .size()
        .reset_index(name="_us_inv_count")
    )

    # ── Acquisitions ───────────────────────────────────────
    acquired = set(acquisitions["company_permalink"].dropna())
    companies["was_acquired"] = companies["permalink"].isin(acquired).astype(int)

    # ── Date features ──────────────────────────────────────
    companies["founded_at"]       = pd.to_datetime(companies["founded_at"],       errors="coerce")
    companies["first_funding_at"] = pd.to_datetime(companies["first_funding_at"], errors="coerce")
    companies["days_to_first_funding"] = (
        companies["first_funding_at"] - companies["founded_at"]
    ).dt.days

    # ── Categorical encodings ──────────────────────────────
    companies["category_code_enc"] = companies["category_code"].astype("category").cat.codes
    companies["country_code_enc"]  = companies["country_code"].astype("category").cat.codes
    companies["state_code_enc"]    = companies["state_code"].astype("category").cat.codes

    # ── Merge everything onto companies ───────────────────
    def _left_merge(base, other, on_left="permalink", on_right="company_permalink"):
        merged = base.merge(other, left_on=on_left, right_on=on_right, how="left")
        if on_right in merged.columns:
            merged.drop(columns=[on_right], inplace=True)
        return merged

    companies = _left_merge(companies, seed_total)
    companies = _left_merge(companies, seed_count)
    companies = _left_merge(companies, other_count)
    companies = _left_merge(companies, investor_count)
    companies = _left_merge(companies, distinct_investors)
    companies = _left_merge(companies, us_investors)

    # ── Derived ratio features ─────────────────────────────
    # avg seed round size  (guard against div-by-zero)
    companies["avg_seed_round_size"] = np.where(
        companies["num_seed_rounds"].fillna(0) > 0,
        companies["total_seed_usd"].fillna(0) / companies["num_seed_rounds"],
        0,
    )
    # investor density per round
    companies["investor_per_round"] = np.where(
        companies["funding_rounds"].fillna(0) > 0,
        companies["num_investors"].fillna(0) / companies["funding_rounds"],
        0,
    )
    # US investor flag
    companies["has_us_investor"] = (companies["_us_inv_count"].fillna(0) > 0).astype(int)

    # ── Assemble & split ───────────────────────────────────
    companies[FEATURE_COLS] = companies[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    X = companies[FEATURE_COLS].copy().fillna(-1).values.astype(float)
    y = companies["reached_series_a"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return X_train, y_train, X_val, y_val, list(FEATURE_COLS)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Feature engineering  (Step 2 in the workflow)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(X_train, X_val, feature_cols, degree=2):
    """
    Expand the *numeric* columns with polynomial + interaction terms,
    then append the categorical columns unchanged.

    Why only numeric columns?
    ─────────────────────────
    PolynomialFeatures on integer-coded categoricals (e.g. category_code_enc=3)
    produces meaningless squared/cross terms.  We therefore split the matrix,
    expand only the numeric block, and re-attach the categorical block.

    Parameters
    ----------
    X_train, X_val  : raw arrays from load_data()
    feature_cols    : list returned by load_data()
    degree          : polynomial degree (default 2; raise to 3 cautiously —
                      feature count explodes as C(n+d, d))

    Returns
    -------
    X_train_eng, X_val_eng  : np.ndarray  (expanded)
    eng_cols                : list[str]   (feature names after expansion)
    """
    n_numeric = feature_cols.index("category_code_enc")   # same as _N_NUMERIC

    # Split numeric / categorical blocks
    X_train_num, X_train_cat = X_train[:, :n_numeric], X_train[:, n_numeric:]
    X_val_num,   X_val_cat   = X_val[:,   :n_numeric], X_val[:,   n_numeric:]

    # Fit polynomial expansion on train only (prevent val leakage)
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_train_poly = poly.fit_transform(X_train_num)
    X_val_poly   = poly.transform(X_val_num)

    # Build readable names for the new columns
    raw_num_names = feature_cols[:n_numeric]
    poly_names    = poly.get_feature_names_out(raw_num_names).tolist()
    cat_names     = feature_cols[n_numeric:]

    # Recombine: [polynomial block | categorical block]
    X_train_eng = np.hstack([X_train_poly, X_train_cat])
    X_val_eng   = np.hstack([X_val_poly,   X_val_cat])
    eng_cols    = poly_names + cat_names

    print(f"[engineer_features] raw features: {len(feature_cols)}  →  "
          f"engineered features: {len(eng_cols)}  (degree={degree})")

    return X_train_eng, X_val_eng, eng_cols


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_val, y_val):
    """Compute validation AUC-ROC (higher is better)."""
    from sklearn.metrics import roc_auc_score
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1]
    else:
        y_score = model.decision_function(X_val)
    return roc_auc_score(y_val, y_score)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Logging
# ══════════════════════════════════════════════════════════════════════════════

def log_result(experiment_id, val_auc, status, variable_changed, value_tested, fixed_conditions, confidence_note):
    """
    Append one row to results.tsv.

    Fields
    ------
    experiment_id    : int — auto-incremented run number (1, 2, 3, ...)
    val_auc          : float — AUC-ROC on validation set
    status           : str — one of: keep, discard, crash
    variable_changed : str — the single variable modified this run
    value_tested     : str — the specific value that was tested
    fixed_conditions : str — all conditions held constant
    confidence_note  : str — why the result is or isn't reliable
    """
    import csv
    RESULTS_FILE = "results.tsv"
    fieldnames = [
        "experiment_id",
        "val_auc",
        "status",
        "variable_changed",
        "value_tested",
        "fixed_conditions",
        "confidence_note",
    ]
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "experiment_id":    experiment_id,
            "val_auc":          round(float(val_auc), 6),
            "status":           status,
            "variable_changed": variable_changed,
            "value_tested":     value_tested,
            "fixed_conditions": fixed_conditions,
            "confidence_note":  confidence_note,
        })
