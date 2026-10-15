"""Compare a fresh run of E1-E4 with the values reported in the paper.

Usage: python3 experiments/check_results.py [paper/generated/summary.json] [results/expected/summary_E1-E4.json]
Exit code 1 if a reported figure differs by more than the tolerance (0.5 point or 0.5 %).
"""
import json, sys

new = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "paper/generated/summary.json"))
ref = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "results/expected/summary_E1-E4.json"))
CHECKS = [("E2", k) for k in ref["E2"]] + [("E2_groups", k) for k in ref.get("E2_groups", {})] + [("E2_used_median", k) for k in ref["E2_used_median"]] \
    + [("E3", k) for k in ref["E3"]] \
    + [("E4", V, f) for V in ref["E4"] for f in ("F1", "fa_P0", "fa_P6", "ko_precision")] \
    + [("E4", V, "det", t) for V in ref["E4"] for t in ref["E4"][V]["det"]] \
    + [("E1", k, "median") for k in ref["E1"] if not k.startswith("t_")]
bad = 0
def get(d, path):
    for p in path: d = d[p]
    return d
for path in CHECKS:
    a, b = get(new, path), get(ref, path)
    tol = max(0.5, 0.005 * abs(b))
    if abs(a - b) > tol:
        bad += 1; print("DIFF", "/".join(path), "obtenu", round(a, 3), "attendu", round(b, 3))
print(f"{len(CHECKS) - bad}/{len(CHECKS)} valeurs conformes à l'article")
sys.exit(1 if bad else 0)
