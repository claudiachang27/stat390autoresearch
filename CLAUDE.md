# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STAT 390 AutoResearch demo: iteratively improve a machine learning model that predicts Series A startup funding success on Crunchbase data. The agent modifies `model.py`, runs experiments, and keeps changes that improve AUC-ROC.

## Commands

```bash
# Run an experiment (logs to results.tsv)
python run.py "description of change"

# Run baseline experiment (forces status="baseline" in log)
python run.py "description" --baseline

# Run and force status="discard" in log regardless of performance
python run.py "description" --discard

# Regenerate performance plot
python prepare.py
```

## Architecture

**Frozen layer** (`prepare.py`): data loading, train/val split (80/20), ROC-AUC evaluation, and plotting. Do not modify. Exposes:
- `load_data()` → `X_train, y_train, X_val, y_val, feature_cols`
- `engineer_features(X_train, X_val, feature_cols, degree=2)` → `X_train_eng, X_val_eng, eng_cols` — expands the 12 numeric columns with polynomial/interaction terms; the 3 categorical columns are passed through unchanged.
- `evaluate(model, X_val, y_val)` → `val_auc` (float) — computes AUC-ROC.
- `log_result(commit, val_auc, status, description)` — appends one row to `results.tsv`.

> **Known issue**: `evaluate` and `log_result` are imported by `run.py` but are not currently defined in `prepare.py`. Running `python run.py` will raise an `ImportError` until these functions are added to `prepare.py`.

**Editable layer** (`model.py`): defines `build_model()`, which must return a sklearn-compatible `Pipeline` or estimator. This is the only file the agent should modify.

**Orchestration** (`run.py`): loads data via `prepare.py`, calls `build_model()`, trains, evaluates, and appends a result row to `results.tsv`. The `experiment` field it logs is the short git commit hash of `HEAD` at run time.

**Demo script** (`demo.py`): standalone simulation using California Housing data — not part of the live experiment workflow.

## Constraints

- Only modify `model.py` — no new files, no changes to `prepare.py` or `run.py`
- Model must be sklearn-compatible (implements `.fit()` / `.predict_proba()`)
- Training + evaluation must complete in under 60 seconds on CPU
- No external data sources or new pip dependencies
- Change only one variable per experiment; hold all others fixed

## Data

15 pre-engineered features from Crunchbase CSVs in `data/` (companies, rounds, investments, acquisitions). Target: binary label (`reached_series_a`). Features include funding totals, seed round counts, investor counts/diversity, acquisition flag, days-to-first-funding, ratio features (`avg_seed_round_size`, `investor_per_round`), a US-investor flag, and three ordinal-encoded categoricals (`category_code_enc`, `country_code_enc`, `state_code_enc`). Missing values are filled with `-1` before the split.

## Experiment Loop

Follow `program.md` for the current experiment queue. The standard per-iteration workflow:

```
1. Read current model.py
2. Propose one modification
3. Edit model.py
4. python run.py "description"
5. Read val_auc from output
6. KEEP  (val_auc > best in results.tsv):  git add model.py && git commit -m "feat: <description>"
7. DISCARD (val_auc ≤ best):               git checkout model.py
8. CRASH (run.py errors / no new row):     log row manually with status="crash", git checkout model.py
9. Repeat
```

Never ask the user for confirmation mid-loop. Make keep/discard decisions autonomously based on `results.tsv`.

## results.tsv Schema

Every run appends exactly one row with these tab-separated fields:

| Field | Notes |
|---|---|
| `experiment` | Short git commit hash of HEAD at run time |
| `val_auc` | AUC-ROC on the validation split |
| `status` | `keep`, `discard`, `baseline`, or `crash` |
| `description` | Free-text description passed to `run.py` |

> **Note**: The file header currently reads `experiment val_rmse val_r2 status description` — a leftover from an earlier regression demo. Ignore the header; the classification rows have 4 fields (`git_hash`, `val_auc`, `status`, `description`). When logging a crash row manually, use 4 tab-separated fields to match this format.
