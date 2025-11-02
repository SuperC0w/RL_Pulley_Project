import argparse
import optuna
import yaml

# === CLI ===
def parse_args():
    p = argparse.ArgumentParser(description="Create scaled RL-Zoo hyperparams from Optuna best.")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Number of vectorized environments to target (scales train_freq, etc.).")
    p.add_argument("--study-name", default="pulley_sac_hpo")
    p.add_argument("--storage", default="sqlite:///pulley_sac_hpo.db")
    p.add_argument("--base-yaml", default="hyperparams/pulley.yaml")
    p.add_argument("--out-yaml", default="hyperparams/pulley_best.yaml")
    return p.parse_args()

args = parse_args()

# === Load Optuna best and map helper params -> SB3 args ===
study = optuna.load_study(study_name=args.study_name, storage=args.storage)
best = study.best_trial.params.copy()
print("Best raw params:", best)

# remove train_freq since we use default to train SAC
del best['train_freq']

# batch_size = 2 ** batch_size_pow
if "batch_size_pow" in best:
    best["batch_size"] = 2 ** int(best.pop("batch_size_pow"))

# gamma = 1 - one_minus_gamma
if "one_minus_gamma" in best:
    best["gamma"] = 1.0 - float(best.pop("one_minus_gamma"))

# net_arch label -> actual list
label = best.get("net_arch")
if isinstance(label, str):
    mapping = {"small": [64, 64], "medium": [256, 256], "big": [400, 300]}
    best["policy_kwargs"] = {"net_arch": mapping[label]}
    best.pop("net_arch", None)

print("Resolved params:", best)

# === Load base yaml and merge ===
with open(args.base_yaml, "r") as f:
    base = yaml.safe_load(f)
cfg = base["PulleyEnv-v0"]
cfg.update(best)

# === Write out YAML ===
out = {"PulleyEnv-v0": cfg}
with open(args.out_yaml, "w") as f:
    yaml.safe_dump(out, f, sort_keys=False)
print(f"Wrote {args.out_yaml}")
