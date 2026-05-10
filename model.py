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

def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000)),
    ])
