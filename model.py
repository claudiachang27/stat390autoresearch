"""
EDITABLE -- The agent modifies this file.
Define the model pipeline for Predicting Series A Startup (classification).
The function build_model() must return an sklearn-compatible estimator.
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def build_model():
    """Return an sklearn Pipeline. This is what the agent improves."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=20,
            class_weight="balanced",   # handles imbalance (most startups don't get Series A)
            random_state=42,
        )),
    ])
