import os
import json

metrics = {}

with open("results/metrics/baseline_metrics.json") as f:
    metrics["baselines_random_split"] = json.load(f)

with open("results/metrics/proposed_piu_metrics.json") as f:
    metrics["proposed_piu_random_split"] = json.load(f)

with open("results/metrics/ablation_metrics.json") as f:
    metrics["ablations_variants_a_to_g"] = json.load(f)

with open("results/metrics/generalization_metrics.json") as f:
    metrics["generalization"] = json.load(f)

with open("results/verification/calibration_verification.json") as f:
    metrics["conformal_calibration"] = json.load(f)

with open("results/verification/bootstrap_verification.json") as f:
    metrics["double_cold_start_bootstraps"] = json.load(f)

with open("results/verification/final_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved combined final metrics to results/verification/final_metrics.json")
