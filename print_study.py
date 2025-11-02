"""
Prints summary statistics for an Optuna study.

Usage:
  python print_study.py --study-name pulley_sac_hpo --storage sqlite:///pulley_sac_hpo.db
Options:
  --top-k N            Show top N completed trials (default: 5)
  --show-failed        List failed trials and their reasons (if saved)
  --csv PATH           Also export a CSV of all trials to PATH
"""

import argparse
from collections import Counter

import optuna

try:
    import pandas as pd  # optional, only needed if --csv is used
except Exception:
    pd = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study-name", required=True)
    p.add_argument("--storage", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--show-failed", action="store_true")
    p.add_argument("--csv", type=str, default=None)
    args = p.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)

    print("=" * 80)
    print(f"Study: {study.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Direction: {study.direction.name}")
    print("=" * 80)

    # Basic counts
    states = Counter(t.state.name for t in study.trials)
    total = len(study.trials)
    finished = sum(t.state.is_finished() for t in study.trials)
    print(f"Total trials: {total}")
    for key in ("COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING"):
        if states.get(key, 0):
            print(f"{key:<8}: {states[key]}")
    print(f"Finished : {finished}")
    print("-" * 80)

    # Best trial (if any complete)
    complete_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    if complete_trials:
        best = study.best_trial
        print("Best trial")
        print(f"  number: {best.number}")
        print(f"  value : {best.value}")
        print(f"  params: {best.params}")
    else:
        print("No COMPLETE trials yet.")
    print("-" * 80)

    # Top-K completed trials
    if complete_trials:
        topk = sorted(complete_trials, key=lambda t: t.value, reverse=(study.direction.name == "MAXIMIZE"))
        topk = topk[: args.top_k]
        print(f"Top {len(topk)} completed trials:")
        for t in topk:
            print(f"  #{t.number:<3d} value={t.value} params={t.params}")
    print("-" * 80)

    # Failed trials with reasons (if available)
    if args.show_failed:
        failed = [t for t in study.trials if t.state.name == "FAIL"]
        if failed:
            print("Failed trials:")
            for t in failed:
                reason = t.system_attrs.get("fail_reason") if isinstance(t.system_attrs, dict) else None
                print(f"  #{t.number:<3d} reason: {reason or 'N/A'} params={t.params}")
        else:
            print("No FAILED trials.")
        print("-" * 80)

    # Optional CSV export
    if args.csv:
        if pd is None:
            print("Pandas not installed; cannot export CSV. Install with `pip install pandas`.")
        else:
            df = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs", "system_attrs"))
            df.to_csv(args.csv, index=False)
            print(f"Wrote CSV to {args.csv}")

    # Small helper: show unresolved helper params (common in rl_zoo3)
    # You can uncomment to see if helper params exist in best trial:
    # if complete_trials:
    #     hp = study.best_trial.params.copy()
    #     if "batch_size_pow" in hp or "one_minus_gamma" in hp or "net_arch" in hp:
    #         print("-" * 80)
    #         print("Note: Your best params include helper variables. You'll likely need to map them:")
    #         print("  batch_size = 2 ** batch_size_pow")
    #         print("  gamma = 1.0 - one_minus_gamma")
    #         print("  net_arch may be a label mapped to a list in your HPO code.")


if __name__ == "__main__":
    main()