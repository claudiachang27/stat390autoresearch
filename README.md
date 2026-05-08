# Series A Success

## Problem

I will be predicting if a start-up will make it to Series A.

**Metric**: AUC-ROC (higher is better)

**Data**: Crunchbase CSVs (no download needed)

## Project Structure

```
predictive-series-a/
├── README.md              # Setup and instructions on how to run agent
├── program.md             # Agent’s Goal: “Predict Series A success with AUC > 0.75
├── prepare.py             # FROZEN: loads crunchbase CSVs, splits data, calculates AUC-ROC
├── model.py               # AGENT-MUTABLE: builds model
├── run.py                 # FROZEN: executes one training loop, appends results to results.tsv
├── results.tsv            # Log of all experiments: timestamp, features used, AUC-ROC
├── performance.png        # Plot showing AUC improving over time (iterations)
├── .gitignore             # Ignore local data files, pycache
└── data/                  # Store Crunchbase CSV files
    └── crunchbase-acquisitions.csv
    └── crunchbase-companies.csv
    └── crunchbase-investments.csv
    └── crunchbase-rounds.csv
```

**Key rule**: the agent may only modify `model.py`. Everything else is frozen.

---

## Setup

### 1. Install an AI coding agent (CLI)

You need a CLI coding agent that can read files, edit files, and run shell commands.
Two recommended options — **neither requires an API key**.

#### Option A: Claude Code CLI (recommended)

```bash
# macOS / Linux / WSL — one-line install
curl -fsSL https://claude.ai/install.sh | bash

# macOS — or via Homebrew
brew install --cask claude-code

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex

# Windows — or via WinGet
winget install Anthropic.ClaudeCode
```

Then launch:

```bash
cd demo_autoresearch
claude
```

First launch opens a browser for login — **no API key needed**.
Works with any Claude subscription (Pro $20/mo, Max, or Team).

Docs: https://code.claude.com/docs

#### Option B: OpenAI Codex CLI

```bash
# Install
npm install -g @openai/codex

# Launch
cd demo_autoresearch
codex
```

First launch opens a browser for ChatGPT login — **no API key needed**.
Works with ChatGPT Plus or higher.

Docs: https://github.com/openai/codex

### 2. Install Python environment

This project requires **Python 3.10+** with `scikit-learn`, `matplotlib`, and `numpy`.

#### Check if Python is installed

```bash
python3 --version
# Should print Python 3.10.x or higher
# If not installed, see below
```

#### Install Python (if needed)

```bash
# macOS
brew install python@3.12

# Ubuntu / Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv

# Windows — download from https://www.python.org/downloads/
# During install, check "Add Python to PATH"
```

#### Install dependencies

```bash
# Option A: with pip (simplest)
pip install scikit-learn matplotlib numpy

# Option B: with uv (faster, used in the main autoresearch project)
# Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install scikit-learn matplotlib numpy

# Option C: with conda
conda install scikit-learn matplotlib numpy
```

#### What gets installed

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | >= 1.3 | ML models, pipelines, evaluation |
| matplotlib | >= 3.7 | Performance plotting |
| numpy | >= 1.24 | Array operations (scikit-learn dependency) |

No GPU, no PyTorch, no heavy downloads — everything runs on CPU.

### 3. Verify setup

```bash
# Quick check: all imports work
python3 -c "import sklearn, matplotlib, numpy; print('All good')"

# Full check: run one experiment
python3 run.py "test run"
# Expected output:
#   Data: 16512 train, 4128 val, 8 features
#   val_rmse: 0.745581
#   val_r2:   0.575788
#   Result logged to results.tsv

# Clean up test result
rm -f results.tsv
```

---

## How to Run the Agent Loop

### Quick start (copy-paste this prompt into your agent)

```
Read program.md for your instructions, then read model.py.
Run `python run.py "baseline"` to establish the baseline RMSE.
Then enter the AutoResearch loop:

1. Propose one modification to model.py (e.g., different estimator,
   feature engineering, hyperparameter change).
2. Edit model.py with your change.
3. Run: python run.py "<short description of what you changed>"
4. Compare the new val_auc to the current best.
   - If improved: KEEP the change, note the new best.
   - If worse: REVERT model.py to the previous version.
5. Repeat from step 1. Try at least 6 different ideas.

After all iterations, run `python prepare.py` to generate performance.png.
Print a summary table of all experiments and which were kept vs discarded.
```

### More specific prompt (if you want to control the search)

```
You are an AutoResearch agent. Read program.md for rules.

Your job: minimize val_rmse on California Housing by modifying model.py.

Constraints:
- model.py must define build_model() returning an sklearn estimator
- Do NOT modify prepare.py or run.py
- Each experiment must finish in < 60 seconds

Search strategy:
1. Start with baseline (LogisticRegression)
2. Try feature engineering (PolynomialFeatures, interactions)
3. Re-test LogisticRegression to see if it closes the gap with non-linear models
4. Try non-linear models (XGBoost and RandomForest) on engineered features
5. Hyperparameter tune the best model

For each experiment:
- Run: python run.py "<description>"
- If val_rmse improved → keep
- If val_rmse worsened → revert model.py to previous version
- Log your reasoning for each decision

After finishing, run: python prepare.py
```

---

## Example Agent Loop (actually executed)

Below is a real agent session. The agent modified `model.py` 7 times,
kept 5 improvements, discarded 1 regression.

### Iteration 0 — Baseline

```python
# model.py
Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
```

```
$ python run.py "baseline: LinearRegression + StandardScaler"
val_rmse: 0.745581   val_r2: 0.5758
```

**BASELINE established**: RMSE = 0.7456

---

### Iteration 1 — Ridge regression

Changed `LinearRegression()` → `Ridge(alpha=1.0)`.

```
$ python run.py "Ridge(alpha=1.0)"
val_rmse: 0.745557   val_r2: 0.5758
```

**KEEP** (marginal improvement). Best = 0.7456

---

### Iteration 2 — Polynomial feature interactions

Added `PolynomialFeatures(degree=2, interaction_only=True)` before Ridge.

```
$ python run.py "PolyFeatures(2, interaction) + Ridge"
val_rmse: 0.703280   val_r2: 0.6226
```

**KEEP** (improved 5.7%). Best = 0.7033

---

### Iteration 3 — Degree-3 polynomials (overshoot)

Changed to `PolynomialFeatures(degree=3)` — full polynomial expansion.

```
$ python run.py "PolyFeatures(3, full) + Ridge -- risky"
val_rmse: 4.880809   val_r2: -17.18
```

**DISCARD** (exploded). Reverted model.py back to iteration 2.

---

### Iteration 4 — Random Forest

Replaced entire pipeline with `RandomForestRegressor(n_estimators=100)`.

```
$ python run.py "RandomForest(n=100)"
val_rmse: 0.505340   val_r2: 0.8051
```

**KEEP** (improved 28%). Best = 0.5053

---

### Iteration 5 — Gradient Boosting

Switched to `GradientBoostingRegressor(n_estimators=200, max_depth=5)`.

```
$ python run.py "GradientBoosting(n=200, depth=5, lr=0.1)"
val_rmse: 0.473611   val_r2: 0.8288
```

**KEEP** (improved 6.3%). Best = 0.4736

---

### Iteration 6 — HistGradientBoosting (tuned)

Switched to `HistGradientBoostingRegressor(max_iter=300, max_depth=8, lr=0.08)`.

```
$ python run.py "HistGBT(iter=300, depth=8, lr=0.08)"
val_rmse: 0.447989   val_r2: 0.8468
```

**KEEP** (improved 5.4%). Best = 0.4480

---

### Summary

| # | Model | RMSE | R² | Decision |
|---|-------|------|----|----------|
| 0 | LinearRegression (baseline) | 0.7456 | 0.576 | baseline |
| 1 | Ridge(alpha=1.0) | 0.7456 | 0.576 | keep |
| 2 | PolyFeatures(2) + Ridge | 0.7033 | 0.623 | keep |
| 3 | PolyFeatures(3) + Ridge | 4.8808 | -17.2 | **discard** |
| 4 | RandomForest(n=100) | 0.5053 | 0.805 | keep |
| 5 | GradientBoosting(n=200) | 0.4736 | 0.829 | keep |
| 6 | HistGBT(n=300, tuned) | **0.4480** | **0.847** | keep |

**Total improvement: 0.7456 → 0.4480 (40% reduction in RMSE)**

---

## Plotting Results

After running experiments:

```bash
python prepare.py
# Generates performance.png from results.tsv
```

This produces a two-panel chart:
- **Top**: validation RMSE over iterations (green = keep, red = discard, blue = baseline)
- **Bottom**: validation R² over iterations
- **Green line**: best-so-far envelope

---

## Adapting This for Your Own Project

To use this structure for a different task:

1. **Replace `prepare.py`** with your own data loading, evaluation metric, and plotting.
   Keep it frozen — the agent must not touch it.

2. **Replace `model.py`** with whatever the agent should modify
   (model definition, hyperparameters, feature engineering, etc.).

3. **Update `program.md`** with your specific rules, constraints, and search ideas.

4. **Update `run.py`** if your training loop is different
   (e.g., PyTorch instead of sklearn).

The key principle: **separate what changes (model.py) from what measures (prepare.py)**.
