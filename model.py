"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Predicting Series A Startup (classification).
The function build_model() must return an sklearn-compatible estimator.
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", missing_values=-1, keep_empty_features=True)),
        ("scaler", RobustScaler()),
        ("model", GradientBoostingClassifier(n_estimators=600, max_depth=3, max_features='sqrt', learning_rate=0.05, min_samples_leaf=4, random_state=42)),
    ])
