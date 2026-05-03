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
8. Only modify one variable value at a time and fix all other conditions.

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

Experiment ID | val_auc | status | variable_changed | value_tested | fixed_conditions | confidence_note

Experiment should be a number (like 1, 2, 3, etc).

val_auc is our pre-defined metric that will be used to determine the status value.

variable_changed should be the variable changed.

value_tested is the new value of the changed variable.

fixed_conditions are the conditions that are held constant.

confidence_note should explain why our result is reliable or not.

Status must be one of: keep, discard, crash

The agent must never skip this step.

## Experiment Queue (Feature Selection Focus)
Run these in order, one at a time. Do not change the model.

Base model (frozen): VotingClassifier HGBC(w=3)+RF(w=1) deep RF

1a. All features, frozen model — run 1
1b. All features, frozen model — run 2  
1c. All features, frozen model — run 3
2. Mutual information top 10
4. Mutual information top 20
5. Mutual information top 30
6. RFE top 10
7. RFE top 20
8. RFE top 30
9. Correlation filter threshold=0.9
10. Correlation filter threshold=0.8
11. L1/Lasso selection (vary regularization strength)
12. Domain subset (founding team, revenue, sector, funding stage signals)

After the full experiment queue is complete, identify the best performing feature selection method and repeat it 3 times to confirm stability.. Do not repeat any other experiments unless a crash occurs.

If all queued experiments are complete, you may vary one feature selection parameter further. Do not change the model architecture.

## Ideas to explore AFTER feature selection is complete
(Do not run these until the experiment queue above is finished)

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
