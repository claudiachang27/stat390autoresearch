"""
Run one experiment: build model, train, evaluate, log result.
Usage:
    python run.py "description"
    python run.py "description" --baseline
    python run.py "description" --discard
"""
import sys
import time
import subprocess
from prepare import load_data, evaluate, log_result

def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"

def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []
    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)
    description = " ".join(description_parts) if description_parts else "experiment"

    # 1. Load data
    X_train, y_train, X_val, y_val, feature_names = load_data()
    print(f"Data: {X_train.shape[0]} train, {X_val.shape[0]} val, {len(feature_names)} features")

    # 2. Build model
    from model import build_model
    model = build_model()
    print(f"Model: {model}")

    # 3. Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    # 4. Evaluate
    val_auc = evaluate(model, X_val, y_val)
    print(f"val_auc: {val_auc:.6f}")

    # 5. Log
    commit = get_git_hash()
    log_result(commit, val_auc, status, description)
    print(f"Result logged to results.tsv (status={status})")

if __name__ == "__main__":
    main()
