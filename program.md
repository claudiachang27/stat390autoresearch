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

## Workflow

```
1. Read program.md for your instructions
2. Read current model.py
3. Propose one modification (e.g., different estimator, feature engineering, hyperparameter change), but you do not need to ask for permission to edit model.py.
4. Edit model.py
5. Run: python run.py "<short description of what you changed>"
6. Compare the new val_auc to the current best.
   - If improved: keep the change, note the new best.
   - If worse: revert model.py to the previous version.
7. Repeat from step 1. Try 3 different ideas.

After all iterations, run `python prepare.py` to generate performance.png.
Print a summary table of all experiments and which were kept vs discarded.
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
Run these in order, one at a time. Do not change the model architecture or weights.

Baseline Consistency (All Features)

1a. Value: All features (Run 1)

1b. Value: All features (Run 2)

1c. Value: All features (Run 3)

Mutual Information (MI) Sensitivity

2a. Variable: MI Feature Count | Value: 5

2b. Variable: MI Feature Count | Value: 10

2c. Variable: MI Feature Count | Value: 15

After the full experiment queue is complete, identify the best performing feature selection method and repeat it 3 times to confirm stability. Do not repeat any other experiments unless a crash occurs.

Once stability runs are logged, stop and wait for further instructions. Do not vary parameters further or change the model architecture.

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
- Do not perform exploratory data analysis or leakage checks
- Do not deviate from the experiment queue
