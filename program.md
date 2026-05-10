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
9. Explore one direction fully before switching — exhaust hyperparameter tuning on the current best model before trying a different model type. Only switch model type when further tuning yields no improvement.

## Workflow

```
1. Read program.md for your instructions
2. Read current model.py
3. Propose one modification (e.g., different estimator, feature engineering, hyperparameter change), but you do not need to ask for permission to edit model.py
4. Edit model.py
5. Run: python run.py "<short description of what you changed>"
6. Compare the new val_auc to the current best:
   - If improved: keep the change, note the new best
   - If worse: revert model.py to the previous version
7. Log the experiment in results.tsv. See "Logging" for more details
8. Repeat from step 1. Try 3 different ideas (thus a total of 3 different experiments)

After all iterations, run `python plot.py` to generate performance.png
Print a summary table of all experiments and which were kept vs discarded
```

## Logging
Every run must produce exactly one new row in results.tsv with fields:

- Experiment ID: should be a number (like 1, 2, 3, etc)
- AUC-ROC: pre-defined metric that will be used to determine the status value
- Status: keep, discard, or crash
- Variable Changed: the variable changed
- Value Tested: new value of the changed variable
- Fixed Conditions: the conditions that are held constant
- Confidence Note: explains why the result is reliable or not

## Keep / Discard / Crash Rule
- KEEP: val_auc strictly greater than the best val_auc in results.tsv → commit
- DISCARD: val_auc less than or equal to best → git checkout model.py, no commit
- CRASH: run.py errors or no new row appears in results.tsv → log a row manually with status = "crash" and the error message, then git checkout model.py

## Ideas to Explore
Only modify one variable at a time.

- Feature selection
- Different classifiers: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- Feature engineering: PolynomialFeatures, interaction terms
- Preprocessing: RobustScaler, QuantileTransformer
- Hyperparameter tuning within the pipeline
- Class imbalance handling: class_weight = 'balanced'

## What NOT to do

- Do not modify `prepare.py` (data split, metric)
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of `build_model()`
- Do not perform exploratory data analysis or leakage checks
