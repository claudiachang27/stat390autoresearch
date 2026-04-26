# AutoResearch Agent Instructions

## Objective

Maximize **AUC-ROC** on the Predicting Series A Startup task.

## Rules

1. You may **ONLY** modify `model.py`
2. `prepare.py` and `run.py` are **FROZEN** — do not touch them
3. `build_model()` must return an sklearn-compatible estimator (Pipeline preferred)
4. Training + evaluation must complete in **under 60 seconds** on CPU
5. No additional data sources or external downloads
6. Do not ask the user for confirmation mid-loop
7. Make all keep/discard decisions autonomously based on results.tsv

## Setup
At the start of each session:
- Select "Yes, allow all edits during this session" when prompted
- Approve bash commands as they appear (Claude Code safety requirement)

## Workflow

```
1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run:  python run.py "description of change"
5. Check val_auc in output
6. If improved:  git add model.py && git commit -m "feat: <description>"
7. If worse:     git checkout model.py   (revert)
8. Repeat from step 1
```

## Keep / Discard / Crash Rule
- KEEP: val_auc strictly greater than the best val_auc in results.tsv → commit
- DISCARD: val_auc less than or equal to best → git checkout model.py, no commit
- CRASH: run.py errors or no new row appears in results.tsv → log a row manually with status="crash" and the error message, then git checkout model.py

## Logging
Every run must produce exactly one new row in results.tsv with fields:
experiment | val_auc | status | description
Status must be one of: keep, discard, crash
The agent must never skip this step.

## Ideas to explore

- Different classifiers: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- Feature engineering: PolynomialFeatures, interaction terms
- Preprocessing: RobustScaler, QuantileTransformer
- Hyperparameter tuning within the pipeline
- Class imbalance handling: class_weight='balanced'

## What NOT to do

- Do not modify `prepare.py` (data split, metric)
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of `build_model()`
