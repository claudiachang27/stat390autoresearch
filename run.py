"""
Run one experiment: build model, train, evaluate, log result.
Usage:
    python run.py --variable "feature selection" --value "mutual info top 10" --fixed "frozen model, seed 42, 80/20 split" --confidence "stable, rerun 1 of 3"
    python run.py --variable "feature selection" --value "all features baseline" --fixed "frozen model, seed 42, 80/20 split" --confidence "baseline confirmation" --status baseline
"""
import sys
import os
import time
from prepare import load_data, evaluate, log_result


def get_next_experiment_id():
    """Read results.tsv and return the next integer experiment ID."""
    RESULTS_FILE = "results.tsv"
    if not os.path.exists(RESULTS_FILE):
        return 1
    import csv
    with open(RESULTS_FILE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        return 1
    # Find the highest existing experiment_id and add 1
    ids = []
    for row in rows:
        try:
            ids.append(int(row.get("experiment_id", 0)))
        except (ValueError, TypeError):
            pass
    return max(ids, default=0) + 1


def parse_args(args):
    """Parse --key value style arguments."""
    parsed = {
        "variable":   None,
        "value":      None,
        "fixed":      None,
        "confidence": None,
        "status":     None,
    }
    i = 0
    while i < len(args):
        key = args[i].lstrip("-")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            parsed[key] = args[i + 1]
            i += 2
        else:
            parsed[key] = True
            i += 1
    return parsed


def main():
    parsed = parse_args(sys.argv[1:])

    variable_changed = parsed["variable"]   or "unspecified"
    value_tested     = parsed["value"]      or "unspecified"
    fixed_conditions = parsed["fixed"]      or "frozen model, seed 42, 80/20 split"
    confidence_note  = parsed["confidence"] or "no confidence note provided"
    status_override  = parsed["status"]     or None  # baseline / discard / None

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

    # 5. Determine status
    experiment_id = get_next_experiment_id()

    if status_override:
        status = status_override
    else:
        # Auto keep/discard: compare against best val_auc in results.tsv
        best_auc = 0.0
        if os.path.exists("results.tsv"):
            import csv
            with open("results.tsv", newline="") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    try:
                        best_auc = max(best_auc, float(row.get("val_auc", 0)))
                    except (ValueError, TypeError):
                        pass
        status = "keep" if val_auc > best_auc else "discard"

    # 6. Log
    log_result(
        experiment_id=experiment_id,
        val_auc=val_auc,
        status=status,
        variable_changed=variable_changed,
        value_tested=value_tested,
        fixed_conditions=fixed_conditions,
        confidence_note=confidence_note,
    )
    print(f"Logged experiment {experiment_id} to results.tsv (status={status})")

    # 7. Print git instructions for the agent
    if status == "keep":
        print(f'ACTION: git add model.py && git commit -m "feat: {value_tested}"')
    else:
        print("ACTION: git checkout model.py")


if __name__ == "__main__":
    main()
