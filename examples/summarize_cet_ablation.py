
import json
import statistics
import sys

reports = []

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        reports.append(json.load(f))

print("| Arm | Test NLL (mean ± SD) | Mean AUC (mean ± SD) |")
print("|---|---:|---:|")

for i, name in enumerate(["A", "B", "C"]):
    nll = [r["ablation"][i]["test_nll"] for r in reports]
    auc = [
        r["ablation"][i]["mean_eventual_auc"]
        for r in reports
    ]

    print(
        f"| {name} | "
        f"{statistics.mean(nll):.3f} ± {statistics.stdev(nll):.3f} | "
        f"{statistics.mean(auc):.3f} ± {statistics.stdev(auc):.3f} |"
    )
