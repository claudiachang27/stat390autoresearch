"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Predicting Series A Startup (classification).
The function build_model() must return an sklearn-compatible estimator.
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ])
