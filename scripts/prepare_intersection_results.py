#!/usr/bin/env python3
"""Prepare experiments/results/ from intersection_comparison/ data.

Uses ONLY the 4163-trace intersection dataset.  No fallback to result-bak.
Files that have no intersection equivalent are intentionally omitted;
generate_latex_data.py handles missing files gracefully.

Run:  python3 scripts/prepare_intersection_results.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INTER = ROOT / "experiments" / "intersection_comparison"
OUT   = ROOT / "experiments" / "results"

OUT.mkdir(exist_ok=True)


def load(path: Path) -> dict | list:
    if not path.exists():
        print(f"  [SKIP] {path.name} not found")
        return {}
    return json.loads(path.read_text())


def save(name: str, data):
    if not data:
        print(f"  [SKIP] {name} — no data")
        return
    p = OUT / name
    p.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  wrote {p.name}")


# ── 1. eitg_results.json  (from 01_eitg_main.json) ──
print("[1] eitg_results.json")
save("eitg_results.json", load(INTER / "01_eitg_main.json"))

# ── 2. baseline_stddevs.json  (from 09_baseline_stddevs.json) ──
print("[2] baseline_stddevs.json")
save("baseline_stddevs.json", load(INTER / "09_baseline_stddevs.json"))

# ── 3. invariance_decoy_results.json  (from 06_invariance_decoy.json) ──
print("[3] invariance_decoy_results.json")
save("invariance_decoy_results.json", load(INTER / "06_invariance_decoy.json"))

# ── 4. adversarial_results.json  (from 05_adversarial.json) ──
print("[4] adversarial_results.json")
save("adversarial_results.json", load(INTER / "05_adversarial.json"))

# ── 5. depth_ablation_results.json  (from 04_depth_ablation.json) ──
print("[5] depth_ablation_results.json")
save("depth_ablation_results.json", load(INTER / "04_depth_ablation.json"))

# ── 6. gnn_comparison_results.json  (needs list→dict conversion) ──
print("[6] gnn_comparison_results.json")
raw_gnn = load(INTER / "02_gnn_comparison.json")
if isinstance(raw_gnn, list) and len(raw_gnn) >= 3:
    gnn_out = {"gnn": raw_gnn[0], "mi": raw_gnn[1], "ig": raw_gnn[2]}
    save("gnn_comparison_results.json", gnn_out)
else:
    save("gnn_comparison_results.json", raw_gnn if raw_gnn else {})

# ── 7. stratified_results.json  (convert from 08_reviewer_response.json) ──
print("[7] stratified_results.json")
rr = load(INTER / "08_reviewer_response.json")
if rr:
    strat = rr.get("stratified", {})

    stratum_map = {
        "All": "all",
        "Medium-URL (middle 33%)": "medium",
        "Clean-URL (top 33%)": "clean",
    }
    view_map = {
        "URL-only (21D)": "url_only",
        "Non-URL views (92D)": "non_url",
        "All 6 views (125D)": "all_views",
    }

    strat_f1 = {}
    for view_label, view_key in view_map.items():
        strat_f1[view_key] = {}
        for stratum_label, stratum_key in stratum_map.items():
            sd = strat.get(stratum_label, {})
            vd = sd.get(view_label, {})
            strat_f1[view_key][stratum_key] = vd.get("f1", 0)

    save("stratified_results.json", {"stratified_f1": strat_f1})
else:
    save("stratified_results.json", {})

# ── 8. dataset_stats.json — generate from intersection data ──
print("[8] dataset_stats.json — from intersection dataset_stats")
ids = load(INTER / "dataset_stats.json")
if ids:
    dataset_stats = {
        "total_samples": ids["total_samples"],
        "phishing_count": ids["phishing_count"],
        "benign_count": ids["benign_count"],
        "source": ids.get("source", {}),
    }
    save("dataset_stats.json", dataset_stats)
else:
    save("dataset_stats.json", {})

# ── 9. comprehensive_agfl.json (from 03_comprehensive_agfl.json) ──
print("[9] comprehensive_agfl.json")
save("comprehensive_agfl.json", load(INTER / "03_comprehensive_agfl.json"))

# ── 10. interpretability_results.json (from 07_interpretability.json) ──
print("[10] interpretability_results.json")
save("interpretability_results.json", load(INTER / "07_interpretability.json"))

# ── 11. baseline_results.json (Phishpedia / PhishIntention on intersection) ──
print("[11] baseline_results.json")
save("baseline_results.json", load(INTER / "baseline_results.json"))

# ── 12. comparison_table.json ──
print("[12] comparison_table.json")
save("comparison_table.json", load(INTER / "comparison_table.json"))

# ── 13. url_adversarial_v3.json (from 11_url_adversarial_v3.json) ──
print("[13] url_adversarial_v3.json")
save("url_adversarial_v3.json", load(INTER / "11_url_adversarial_v3.json"))

# ── NOTE: The following files have NO intersection equivalent.
# They are intentionally omitted. generate_latex_data.py will skip
# the corresponding sections when these files are missing.
#
# Omitted (no intersection data):
#   - itg_quality_results.json     (ITG quality gap — from old raw crawl)
#   - ppci_validation_results.json (PPCI — from old dataset)
#   - feature_importance.json      (individual features — from old dataset)
#   - path_explosion_results.json  (structural — from old dataset)
#   - runtime_analysis.json        (system timing — from old dataset)
#   - wild_detection_results.json  (separate wild experiment)
#   - visual_baselines_analysis.json (replaced by baseline_results.json)

print("\nDone. Now run: python3 scripts/generate_latex_data.py")
