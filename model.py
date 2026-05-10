"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Predicting Series A Startup (classification).
The function build_model() must return an sklearn-compatible estimator.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Indices of leaking features in FEATURE_COLS:
# 0: funding_total_usd
# 1: funding_rounds
# 6: was_acquired
# 10: investor_per_round (derived from funding_rounds)
LEAKY_INDICES = [0, 1, 6, 10]

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, indices):
        self.indices = indices
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        keep = [i for i in range(X.shape[1]) if i not in self.indices]
        return X[:, keep]

def build_model():
    return Pipeline([
        ("drop_leaky", DropColumns(LEAKY_INDICES)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000)),
    ])
