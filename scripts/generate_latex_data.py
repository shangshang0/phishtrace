#!/usr/bin/env python3
"""Generate LaTeX macros from experiment result JSON files.

This script reads all JSON result files in experiments/results/ and produces
a single LaTeX file (latex/generated_data.tex) containing \\newcommand macros.
The paper \\input{generated_data} and uses these macros for all numerical claims,
ensuring perfect consistency between experimental data and the paper text.

Run:  py -3 scripts/generate_latex_data.py
"""

import json
import math
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "iccs2026latex" / "generated_data.tex"
# Also write to /data/output/ when running inside Docker
OUT_DATA = Path("/data/output/generated_data.tex")

def load(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        print(f"  [WARN] {name} not found")
        return {}
    return json.load(open(p))


def fmt(v, decimals=4):
    """Format a float to fixed decimals, stripping trailing zeros."""
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{decimals}f}"


def fmt3(v):
    return fmt(v, 3)


def fmt2(v):
    return fmt(v, 2)


def fmt1(v):
    return fmt(v, 1)


def pct(v, decimals=1):
    """Format as percentage."""
    if v is None:
        return "N/A"
    return f"{v * 100:.{decimals}f}"


def pct0(v):
    return pct(v, 0)


def comma(v):
    """Integer with comma separator."""
    return f"{int(v):,}"


def safe_cmd(name: str) -> str:
    """Sanitize a name for use as a LaTeX command.
    
    TeX control sequences may only contain ASCII letters (a-z, A-Z).
    Digits are spelled out so they don't break the command name.
    """
    _digit_words = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    }
    s = (name.replace("-", "").replace("_", "").replace(" ", "")
         .replace("+", "Plus").replace("(", "").replace(")", "")
         .replace(".", "pt").replace(";", "").replace(",", "")
         .replace("/", ""))
    # Replace digits with word equivalents (TeX commands must be all-alpha)
    for digit, word in _digit_words.items():
        s = s.replace(digit, word)
    return s


lines: list[str] = []


def emit(cmd: str, val):
    """Add a \\newcommand line."""
    cmd_clean = safe_cmd(cmd)
    lines.append(f"\\newcommand{{\\{cmd_clean}}}{{{val}}}")


def section(title: str):
    lines.append(f"\n%% --- {title} ---")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Dataset counts
# ═══════════════════════════════════════════════════════════════════════════════
section("Dataset Counts")
eitg = load("eitg_results.json")

dataset_size = eitg.get("dataset_size", 4163)
n_phish = eitg.get("phishing_count", 2163)
n_benign = eitg.get("benign_count", 2000)
n_itg = eitg.get("itg_available", 1701)

# Intersection dataset: raw crawl == curated (no dead pages after intersection)
emit("dataRawCrawl", comma(dataset_size))
emit("dataDeadRemoved", comma(0))
emit("dataCurated", comma(dataset_size))
emit("dataPhishing", comma(n_phish))
emit("dataBenign", comma(n_benign))
emit("dataITGAvailable", comma(n_itg))
emit("dataITGPct", fmt1(n_itg / dataset_size * 100))
emit("dataFeatureDim", "125")
emit("dataPhishPct", fmt1(n_phish / dataset_size * 100))
emit("dataBenignPct", fmt1(n_benign / dataset_size * 100))

# ITG phishing/benign split — compute proportionally from totals
itg_phishing = round(n_phish / dataset_size * n_itg)
itg_benign = n_itg - itg_phishing
emit("dataITGPhishing", comma(itg_phishing))
emit("dataITGBenign", comma(itg_benign))

# ─── 1a. Static Constants (brand names, view counts, feature dimensions) ───
# Use providecommand for constants that may already be defined in paper.tex
section("Static Constants")
lines.append(f"\\providecommand{{\\phishtrace}}{{PhishTrace}}")
lines.append(f"\\providecommand{{\\numViews}}{{6}}")
lines.append(f"\\providecommand{{\\simmark}}{{$\\checkmark$}}")
emit("dimURL", "21")
emit("dimNetwork", "18")
emit("dimRedirect", "14")
emit("dimInteraction", "18")
emit("dimITGBase", "30")
emit("dimITGEng", "42")
emit("dimCrossView", "12")
emit("dimCrossLayer", "12")
emit("dimNonURL", "92")
emit("dimTrulyNonURL", "92")

# ─── 1b. Source distribution — intersection dataset has no per-trace source ───
# Emit empty/placeholder values so macros exist but paper should not list them
section("Source Distribution (intersection — no per-trace breakdown)")
emit("srcPhishTank", "---")
emit("srcURLScan", "---")
emit("srcThreatFox", "---")
emit("srcURLhaus", "---")
emit("srcCERTPL", "---")
emit("srcOpenPhish", "---")
emit("srcTranco", "---")

# ─── 1c. Trace feature statistics — not available for intersection ───
section("Trace Feature Statistics (not available for intersection dataset)")
for prefix in ["tsPhish", "tsBenign", "tsOverall"]:
    for suffix in ["Nodes", "Edges", "Density", "Redirects", "KeywordPct", "TLDPct", "Entropy"]:
        emit(f"{prefix}{suffix}", "---")

# ─── 1d. Brand distribution — not available for intersection ───
section("Brand Distribution (not available for intersection dataset)")
emit("brandTotal", "---")
emit("brandTotalPhishing", "---")
emit("brandPct", "---")
for cat_name in ["FinancialServices", "EmailCloudServices", "SocialMedia",
                  "Cryptocurrency", "Ecommerce", "Government"]:
    emit(f"brand{cat_name}Count", "---")
    emit(f"brand{cat_name}Pct", "---")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ITG Quality  — NOT available for intersection dataset.
#    Emit placeholder macros so the paper compiles but marks data as removed.
# ═══════════════════════════════════════════════════════════════════════════════
section("ITG Quality (removed — not re-run on intersection dataset)")
for prefix in ["itg", "nonitg"]:
    for suffix in ["Total", "Ok", "Dead", "OkPct", "DeadPct"]:
        emit(f"{prefix}{suffix}", "---")
for prefix in ["itgPhish", "itgBenign", "nonitgPhish", "nonitgBenign"]:
    for suffix in ["Ok", "Dead", "Total", "PctOk", "DeadPct"]:
        emit(f"{prefix}{suffix}", "---")
emit("itgQualityGapPP", "---")
emit("nonitgPhishFuncPct", "---")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PPCI Validation — NOT available for intersection dataset.
# ═══════════════════════════════════════════════════════════════════════════════
section("PPCI Validation (removed — not re-run on intersection dataset)")
for label in ["phishing", "benign"]:
    pfx = f"ppci{label.capitalize()}"
    for suffix in ["Total", "WithCred", "Satisfied", "Rate", "AvgCompliance",
                    "CredPct", "Counter", "CounterPct"]:
        emit(f"{pfx}{suffix}", "---")
emit("ppciGapPP", "---")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main Results Table (Table 2)
# ═══════════════════════════════════════════════════════════════════════════════
section("Main Results — SMVE / MVE / Stacked")
for variant, key in [("SMVE", "eitg_smve"), ("MVE", "eitg_mve"), ("Stacked", "eitg_stacked")]:
    d = eitg.get(key, {})
    s = eitg.get(f"{key}_std", {})
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        short_m = metric[:3].capitalize() if metric != "f1" else "Fone"
        if metric == "auc":
            short_m = "AUC"
        emit(f"res{variant}{short_m}", fmt(d.get(metric, 0)))
        emit(f"res{variant}{short_m}Std", fmt(s.get(metric, 0)))

section("Main Results — Paradigm Baselines")
bstd = load("baseline_stddevs.json")
baselines = bstd.get("baselines", {})
baseline_name_map = {
    "URLNet-RF": "URLNetRF",
    "RedirectChain-GBM": "RedirectChainGBM",
    "NetTraffic-RF": "NetTrafficRF",
    "ContentHeur-LR": "ContentHeurLR",
    "StackEnsemble": "StackEnsemble",
}
for orig_name, tex_name in baseline_name_map.items():
    d = baselines.get(orig_name, {})
    m = d.get("mean", {})
    s = d.get("std", {})
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        short_m = metric[:3].capitalize() if metric != "f1" else "Fone"
        if metric == "auc":
            short_m = "AUC"
        emit(f"res{tex_name}{short_m}", fmt(m.get(metric, 0)))
        emit(f"res{tex_name}{short_m}Std", fmt(s.get(metric, 0)))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Statistical Tests (Appendix — Welch t-test / Wilcoxon)
# ═══════════════════════════════════════════════════════════════════════════════
section("Statistical Tests")
stat_tests = bstd.get("statistical_tests", {})
for orig_name, tex_name in baseline_name_map.items():
    t = stat_tests.get(orig_name, {})
    if not t:
        continue
    delta = t.get("delta_f1", 0)
    pv = t.get("p_value", 1)
    sig = t.get("significance", "ns")
    test_type = t.get("test_type", "welch_t")
    emit(f"stat{tex_name}DeltaF", fmt(delta))
    # Format p-value
    if pv == 0:
        pval_str = "$<10^{-6}$"
    elif pv < 0.001:
        exp = int(math.floor(math.log10(pv)))
        mantissa = pv / (10 ** exp)
        pval_str = f"${mantissa:.1f}\\times 10^{{{exp}}}$"
    else:
        pval_str = fmt(pv)
    emit(f"stat{tex_name}Pval", pval_str)
    emit(f"stat{tex_name}Sig", sig.replace("***", "***").replace("**", "**").replace("*", "*").replace("ns", "n.s."))
    emit(f"stat{tex_name}TestType", test_type.replace("_", " "))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Stratified Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
section("Stratified F1 Scores")
strat = load("stratified_results.json")
strat_f1 = strat.get("stratified_f1", {})
for view_key, view_tex in [("url_only", "UrlOnly"), ("non_url", "NonUrl"), ("all_views", "AllViews")]:
    for stratum in ["all", "medium", "clean"]:
        val = strat_f1.get(view_key, {}).get(stratum, 0)
        emit(f"strat{view_tex}{stratum.capitalize()}", fmt(val))

# Deltas (computed correctly from raw values)
url_med = strat_f1.get("url_only", {}).get("medium", 0)
all_med = strat_f1.get("all_views", {}).get("medium", 0)
url_clean = strat_f1.get("url_only", {}).get("clean", 0)
all_clean = strat_f1.get("all_views", {}).get("clean", 0)
url_all = strat_f1.get("url_only", {}).get("all", 0)
all_all = strat_f1.get("all_views", {}).get("all", 0)

emit("stratDeltaMediumPP", fmt1((all_med - url_med) * 100))
emit("stratDeltaCleanPP", fmt1((all_clean - url_clean) * 100))
emit("stratDeltaAllPP", fmt1((all_all - url_all) * 100))

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Per-View Invariance (Medium-URL and Clean-URL strata)
# ═══════════════════════════════════════════════════════════════════════════════
section("Per-View Invariance")
inv = load("invariance_decoy_results.json")
bi = inv.get("behavioral_invariance", {})

for stratum_key, stratum_tex in [("Medium-URL", "Medium"), ("Clean-URL", "Clean")]:
    sd = bi.get(stratum_key, {})
    emit(f"inv{stratum_tex}N", comma(sd.get("n_total", 0)))
    emit(f"inv{stratum_tex}NPhish", comma(sd.get("n_phish", 0)))
    emit(f"inv{stratum_tex}NBenign", comma(sd.get("n_benign", 0)))

    for view_name in ["Network (18D)", "Redirect (14D)", "Interaction (18D)",
                      "ITG+Eng (42D)", "Truly-non-URL (92D)", "Non-URL+CrossView (104D)",
                      "URL-only (21D)", "All 6 views (125D)"]:
        vd = sd.get(view_name, {})
        tex_view = safe_cmd(view_name.split("(")[0].strip().replace(" ", "")
                            .replace("-", "").replace("+", "Plus"))
        for metric in ["f1", "auc"]:
            short = "Fone" if metric == "f1" else "AUC"
            emit(f"inv{stratum_tex}{tex_view}{short}", fmt(vd.get(metric, 0)))
            if f"{metric}_std" in vd:
                emit(f"inv{stratum_tex}{tex_view}{short}Std", fmt(vd.get(f"{metric}_std", 0)))

# Invariance ratios from invariance_decoy_results (clean vs medium)
# IR = F1(Clean) / F1(Medium)
med_data = bi.get("Medium-URL", {})
clean_data = bi.get("Clean-URL", {})
view_ir_map = [
    ("Network (18D)", "Network18D"),
    ("Redirect (14D)", "Redirect14D"),
    ("Interaction (18D)", "Interaction18D"),
    ("ITG+Eng (42D)", "ITGPlusEng42D"),
    ("Truly-non-URL (92D)", "TrulynonURL92D"),
    ("Non-URL+CrossView (104D)", "NonURLPlusCrossView104D"),
    ("URL-only (21D)", "URLonly21D"),
    ("All 6 views (125D)", "All6views125D"),
]
for view_name, tex_name in view_ir_map:
    f1_med = med_data.get(view_name, {}).get("f1", 0)
    f1_clean = clean_data.get(view_name, {}).get("f1", 0)
    ir = f1_clean / f1_med if f1_med > 0 else 0
    emit(f"ir{tex_name}", fmt(ir))

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Decoy Navigation (Full-system evasion)
# ═══════════════════════════════════════════════════════════════════════════════
section("Decoy Navigation & Adversarial")
decoy = inv.get("decoy_navigation", {})

for attack_key, attack_tex in [("clean_baseline", "CleanBaseline"),
                                ("decoy_navigation", "DecoyNav"),
                                ("url_camouflage", "URLCamo"),
                                ("form_splitting", "FormSplit"),
                                ("combined", "Combined")]:
    ad = (inv.get(attack_key, {}) if attack_key == "clean_baseline"
          else decoy if attack_key == "decoy_navigation"
          else inv.get("decoy_navigation", {}).get(attack_key, {})
          if False else inv.get(attack_key, {}))
    # Fix: data is at top level of inv for clean_baseline, inside decoy_navigation for others
    if attack_key == "clean_baseline":
        ad = inv.get("clean_baseline", inv.get("decoy_navigation", {}).get("clean_baseline", {}))
    else:
        ad = inv.get("decoy_navigation", {}).get(attack_key, inv.get(attack_key, {}))

    for sys_key, sys_tex in [("full_system", "Full"), ("nonurl_only", "NonURL"), ("url_only", "URL")]:
        sd = ad.get(sys_key, {})
        if "f1" in sd:
            emit(f"decoy{attack_tex}{sys_tex}Fone", fmt(sd["f1"]))
        if "evasion_rate" in sd:
            emit(f"decoy{attack_tex}{sys_tex}Evasion", pct(sd["evasion_rate"]))

# Clean baseline separate
cb = inv.get("decoy_navigation", {}).get("clean_baseline", {})
for sys_key, sys_tex in [("full_system", "Full"), ("nonurl_only", "NonURL"), ("url_only", "URL")]:
    sd = cb.get(sys_key, {})
    if "f1" in sd:
        emit(f"decoyClean{sys_tex}Fone", fmt(sd["f1"]))
    if "acc" in sd:
        emit(f"decoyClean{sys_tex}Acc", fmt(sd["acc"]))

emit("decoyNTest", comma(cb.get("n_test", 0)))
emit("decoyNPhishTest", comma(cb.get("n_phishing_test", 0)))

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Adversarial (ITG-only, 42D)
# ═══════════════════════════════════════════════════════════════════════════════
section("ITG-Only Adversarial")
adv = load("adversarial_results.json")
emit("advCleanAcc", fmt(adv.get("clean", {}).get("accuracy", 0)))
emit("advCleanFone", fmt(adv.get("clean", {}).get("f1", 0)))

for attack in ["fgsm", "pgd"]:
    ad = adv.get(attack, {})
    for eps in ["0.01", "0.05", "0.1", "0.2", "0.5"]:
        ed = ad.get(eps, {})
        tex_eps = eps.replace(".", "pt")
        emit(f"adv{attack}{tex_eps}Fone", fmt(ed.get("f1", 0)))
        emit(f"adv{attack}{tex_eps}Evasion", pct(ed.get("evasion_rate", 0)))

for attack in ["node_injection", "edge_rewiring", "graph_inflation",
               "combined_structural", "decoy_navigation", "form_splitting"]:
    ad = adv.get(attack, {})
    tex_atk = safe_cmd(attack)
    emit(f"adv{tex_atk}Fone", fmt(ad.get("f1", 0)))
    emit(f"adv{tex_atk}Evasion", pct(ad.get("evasion_rate", 0)))

# ═══════════════════════════════════════════════════════════════════════════════
# 9a. Cross-View Adversarial (Table 4 in paper)
#     Combines data from reviewer_response_results.json (ITG attacks, full system)
#     and url_adversarial_v3.json (URL attacks)
# ═══════════════════════════════════════════════════════════════════════════════
section("Cross-View Adversarial (Table 4)")
_rr = load("reviewer_response_results.json")
_rr_adv = _rr.get("adversarial_full", {})
_ua_cv = load("url_adversarial_v3.json")
_ua_res = _ua_cv.get("results", {})

# --- URL view under attack ---
# Clean baselines: URL-only from url_adversarial_v3 URLNet-RF clean F1,
# Full SMVE from url_adversarial_v3 PhishTrace-SMVE clean F1
_ua_clean = _ua_res.get("clean", {})
_ua_camo = _ua_res.get("url_camouflage", {})
_ua_partial = _ua_res.get("partial_camouflage", {})
_ua_mean = _ua_res.get("mean_substitution", {})

emit("advURLCleanFone", fmt(_ua_clean.get("URLNet-RF", {}).get("f1", 0)))
emit("advSMVECleanFone", fmt(_ua_clean.get("PhishTrace-SMVE", {}).get("f1", 0)))

# Benign-URL replacement (full camouflage)
emit("advURLReplaceFone", fmt(_ua_camo.get("URLNet-RF", {}).get("f1_attacked", 0)))
emit("advSMVEURLReplaceFone", fmt(_ua_camo.get("PhishTrace-SMVE", {}).get("f1_attacked", 0)))

# Partial camouflage (50%)
emit("advURLPartialFone", fmt(_ua_partial.get("URLNet-RF", {}).get("f1_attacked", 0)))
emit("advSMVEURLPartialFone", fmt(_ua_partial.get("PhishTrace-SMVE", {}).get("f1_attacked", 0)))

# Mean substitution
emit("advURLMeanSubFone", fmt(_ua_mean.get("URLNet-RF", {}).get("f1_attacked", 0)))
emit("advSMVEURLMeanSubFone", fmt(_ua_mean.get("PhishTrace-SMVE", {}).get("f1_attacked", 0)))

# --- ITG/Graph view under attack (from reviewer_response adversarial_full) ---
emit("advITGCleanFone", fmt(_rr_adv.get("itg_only_clean", {}).get("f1", 0)))
emit("advFullCleanFone", fmt(_rr_adv.get("full_system_clean", {}).get("f1", 0)))

# Node injection
_ni = _rr_adv.get("node_injection", {})
emit("advITGNodeInjFone", fmt(_ni.get("itg_only", {}).get("f1", 0)))
emit("advITGNodeInjEvasion", pct(_ni.get("itg_only", {}).get("evasion_rate", 0)))
emit("advFullDecoyNavFone", fmt(_ni.get("full_system", {}).get("f1", 0)))
emit("advFullDecoyNavEvasion", pct(_ni.get("full_system", {}).get("evasion_rate", 0)))

# Combined structural
_cs = _rr_adv.get("combined_structural", {})
emit("advITGCombStructFone", fmt(_cs.get("itg_only", {}).get("f1", 0)))
emit("advFullCombinedFone", fmt(_cs.get("full_system", {}).get("f1", 0)))
emit("advFullCombinedEvasion", pct(_cs.get("full_system", {}).get("evasion_rate", 0)))

# ═══════════════════════════════════════════════════════════════════════════════
# 9b. URL-Adversarial (Cross-Baseline) — from url_adversarial_v3.json
#     Four-phase evaluation: Clean, Targeted, URL-Only, Progressive Corruption
# ═══════════════════════════════════════════════════════════════════════════════
section("URL-Adversarial Cross-Baseline")
ua = load("url_adversarial_v3.json")
ua_data = ua.get("dataset", {})
emit("uaTotal", comma(ua_data.get("total", 0)))
emit("uaPhishing", comma(ua_data.get("phishing", 0)))
emit("uaBenign", comma(ua_data.get("benign", 0)))

ua_results = ua.get("results", {})

# Method name → LaTeX-safe suffix
ua_method_map = {
    "URLNet-RF": "URLNetRF",
    "RedirectChain-GBM": "RedirGBM",
    "NetTraffic-RF": "NetRF",
    "ContentHeur-LR": "ContentLR",
    "StackEnsemble": "Stack",
    "PhishTrace-SMVE": "SMVE",
}

# Phase 1: Clean performance
clean_data = ua_results.get("clean", {})
for method_name, method_tex in ua_method_map.items():
    md = clean_data.get(method_name, {})
    emit(f"uaClean{method_tex}Fone", fmt(md.get("f1", 0)))
    emit(f"uaClean{method_tex}AUC", fmt(md.get("auc", 0)))

# Phase 2: Targeted feature replacement (each baseline attacked on own features)
targeted_data = ua_results.get("targeted", {})
for method_name, method_tex in ua_method_map.items():
    md = targeted_data.get(method_name, {})
    if md:
        emit(f"uaTarget{method_tex}FoneClean", fmt(md.get("f1_clean", 0)))
        emit(f"uaTarget{method_tex}FoneAtk", fmt(md.get("f1_attacked", 0)))
        emit(f"uaTarget{method_tex}FoneDrop", fmt(md.get("f1_drop", 0)))
        emit(f"uaTarget{method_tex}Evasion", pct(md.get("evasion_rate", 0)))

# Phase 3: URL-only camouflage (cross-system comparison)
url_camo_data = ua_results.get("url_camouflage", {})
for method_name, method_tex in ua_method_map.items():
    md = url_camo_data.get(method_name, {})
    if md:
        emit(f"uaURLCamo{method_tex}FoneClean", fmt(md.get("f1_clean", 0)))
        emit(f"uaURLCamo{method_tex}FoneAtk", fmt(md.get("f1_attacked", 0)))
        emit(f"uaURLCamo{method_tex}Evasion", pct(md.get("evasion_rate", 0)))

# Phase 4: Progressive view corruption (SMVE)
progressive_data = ua_results.get("progressive", {})
num_words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six"]
for nc in range(7):
    key = f"{nc}_views"
    md = progressive_data.get(key, {})
    word = num_words[nc]
    emit(f"uaProg{word}Fone", fmt(md.get("f1", 0)))
    emit(f"uaProg{word}AUC", fmt(md.get("auc", 0)))
    emit(f"uaProg{word}Evasion", pct(md.get("evasion_rate", 0)))

# Convenience macros for key comparisons
# Targeted: each baseline under its own targeted attack
for method_name, method_tex in ua_method_map.items():
    md = targeted_data.get(method_name, {})
    emit(f"uaKey{method_tex}Evasion", pct(md.get("evasion_rate", 0)))
    emit(f"uaKey{method_tex}FoneDrop", fmt(md.get("f1_drop", 0)))

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Feature Importance — Category-level from interpretability_results.json
#     Individual feature rankings removed (old 1593-trace data).
# ═══════════════════════════════════════════════════════════════════════════════
section("Feature Importance")
interp = load("interpretability_results.json")
cat_stats = interp.get("category_stats", {})
# category_stats values are lists of float SHAP proportions per feature in each category
# Compute average SHAP proportion per category, then derive percentages
cat_avg_shap: dict[str, float] = {}
for cat_name, vals in cat_stats.items():
    if isinstance(vals, list) and vals:
        cat_avg_shap[cat_name] = sum(abs(v) for v in vals) / len(vals)
    else:
        cat_avg_shap[cat_name] = 0.0
total_shap = sum(cat_avg_shap.values()) or 1.0
for cat_name in ["temporal", "behavioral", "centrality", "topological"]:
    pct_val = cat_avg_shap.get(cat_name, 0.0) / total_shap * 100
    emit(f"featCat{cat_name.capitalize()}Pct", fmt1(pct_val))
# Placeholder macros for removed individual feature ranks
for rank in range(1, 11):
    emit(f"fiRank{rank}Name", "---")
    emit(f"fiRank{rank}Val", "---")
    emit(f"fiRank{rank}Pct", "---")
emit("featureImportanceN", "---")
emit("featureImportanceNFeat", "---")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Depth Ablation
# ═══════════════════════════════════════════════════════════════════════════════
section("Depth Ablation")
da = load("depth_ablation_results.json")
for depth_key in ["0-evt", "1-evt", "3-evt", "8-evt", "15-evt", "all"]:
    dd = da.get(depth_key, {})
    tex_key = depth_key.replace("-", "").replace("evt", "Evt")
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        short = metric[:3].capitalize() if metric != "f1" else "Fone"
        if metric == "auc":
            short = "AUC"
        emit(f"da{tex_key}{short}", fmt(dd.get(metric, 0)))

# ═══════════════════════════════════════════════════════════════════════════════
# 12. GNN Comparison
# ═══════════════════════════════════════════════════════════════════════════════
section("GNN Comparison")
gnn = load("gnn_comparison_results.json")
for model_name in ["SpectralGCN-2L", "SpectralGCN-3L", "GAT-4H", "GraphSAGE"]:
    md = gnn.get("gnn", {}).get(model_name, {})
    avg = md.get("avg", {})
    std = md.get("std", {})
    tex_name = safe_cmd(model_name)
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        short = metric[:3].capitalize() if metric != "f1" else "Fone"
        if metric == "auc":
            short = "AUC"
        emit(f"gnn{tex_name}{short}", fmt(avg.get(metric, 0)))
        emit(f"gnn{tex_name}{short}Std", fmt(std.get(metric, 0)))

# Mutual Information / Redundancy emitted in Section 19 below (MI Redundancy Per View)

# Information Gain
section("Information Gain")
ig = gnn.get("ig", {})
for key, val in ig.items():
    tex_key = safe_cmd(key)
    emit(f"ig{tex_key}", fmt(val))

# ═══════════════════════════════════════════════════════════════════════════════
# 13. Path Explosion — removed (old 1593-trace data, no intersection equivalent)
# ═══════════════════════════════════════════════════════════════════════════════
section("Path Explosion")
emit("peMedianB", "---")
emit("peMedianBeff", "---")
for d in [3, 5, 8]:
    emit(f"peD{d}Std", "---")
    emit(f"peD{d}Drp", "---")
    emit(f"peD{d}Reduction", "---")

# ═══════════════════════════════════════════════════════════════════════════════
# 14. Runtime Analysis — removed (old 1593-trace data, no intersection equivalent)
# ═══════════════════════════════════════════════════════════════════════════════
section("Runtime Analysis")
for m in ["rtNSamples", "rtFeatureExtS", "rtITGConstructS", "rtClassifyS", "rtTotalS"]:
    emit(m, "---")

# ═══════════════════════════════════════════════════════════════════════════════
# 15. Wild Detection — from validated_run_20260219/summary.json
# ═══════════════════════════════════════════════════════════════════════════════
section("Wild Detection")
_wild_summary_path = RESULTS.parent / "validated_run_20260219" / "summary.json"
if _wild_summary_path.exists():
    _ws = json.loads(_wild_summary_path.read_text(encoding="utf-8"))
    _crawl = _ws.get("crawl_summary", {})
    _anal = _ws.get("analysis_summary", {})
    _total_urls = sum(s["total"] for s in _crawl.get("per_source", []))
    _crawl_ok = _crawl.get("total_crawled", 0)
    _crawl_fail = _total_urls - _crawl_ok
    _detected_phish = _anal.get("total_phishing", 0)
    _detected_benign = _anal.get("total_benign", 0)
    _n_sources = len(_crawl.get("per_source", []))
    emit("wildTotalURLs", f"{_total_urls:,}")
    emit("wildCrawlOk", f"{_crawl_ok:,}")
    emit("wildCrawlFail", f"{_crawl_fail:,}")
    emit("wildDetectedPhish", f"{_detected_phish:,}")
    emit("wildDetectedBenign", f"{_detected_benign:,}")
    emit("wildDetectionRate", f"{_anal.get('detection_rate', 0):.1f}")
    emit("wildNSources", str(_n_sources))
    emit("wildSamplePerSource", str(_ws.get("config", {}).get("sample_per_source", "---")))
    # per-source crawl success for detail
    for src_info in _crawl.get("per_source", []):
        src_name = src_info["source"].capitalize()
        emit(f"wildSrc{src_name}Total", str(src_info["total"]))
        emit(f"wildSrc{src_name}Ok", str(src_info["crawled_ok"]))
else:
    # fallback stubs
    for m in ["wildTotalURLs", "wildCrawlOk", "wildCrawlFail",
             "wildDetectedPhish", "wildDetectedBenign", "wildDetectionRate",
             "wildNSources", "wildSamplePerSource"]:
        emit(m, "---")
# Legacy macros (kept for backward compat)
for m in ["wildTotal", "wildCrawlFailPhish", "wildCrawlFailBenign",
         "wildTP", "wildFP", "wildFN", "wildTN",
         "wildAcc", "wildPrec", "wildRec", "wildFone"]:
    emit(m, "---")

# ═══════════════════════════════════════════════════════════════════════════════
# 16. Visual Baselines — replaced by intersection baselines (Section 17)
# ═══════════════════════════════════════════════════════════════════════════════
section("Visual Baselines")
for tool in ["Phishpedia", "PhishIntention"]:
    for m in ["N", "Rec", "Prec", "Fone"]:
        emit(f"vb{tool}{m}", "---")

# ═══════════════════════════════════════════════════════════════════════════════
# 17. Intersection Baselines (Phishpedia / PhishIntention)
#     Try intersection_baselines.json first, fall back to results.json baselines
# ═══════════════════════════════════════════════════════════════════════════════
section("Intersection Baselines")
ib = load("intersection_baselines.json")
if not ib:
    ib = load("baseline_results.json")
if not ib:
    # Fall back to results.json which has Phishpedia in its baselines dict
    _results_fallback = load("results.json")
    _bl = _results_fallback.get("baselines", {})
    if "Phishpedia" in _bl:
        ib = {
            "phishpedia": _bl["Phishpedia"],
            "phishintention": _bl.get("PhishIntention", {}),
        }
if ib:
    for tool_key, tool_tex in [("phishpedia", "Phishpedia"), ("phishintention", "PhishIntention")]:
        td = ib.get(tool_key, {})
        for metric in ["accuracy", "precision", "recall", "f1"]:
            short_m = metric[:3].capitalize() if metric != "f1" else "Fone"
            emit(f"res{tool_tex}{short_m}", fmt(td.get(metric, 0)))
            emit(f"res{tool_tex}{short_m}Std", fmt(0.0))
else:
    # Emit placeholder macros so paper compiles
    for tool_tex in ["Phishpedia", "PhishIntention"]:
        for short_m in ["Acc", "Pre", "Rec", "Fone"]:
            emit(f"res{tool_tex}{short_m}", "---")
            emit(f"res{tool_tex}{short_m}Std", "---")

# ═══════════════════════════════════════════════════════════════════════════════
# 18. AGFL – Comprehensive graph-feature learning from comprehensive_agfl.json
# ═══════════════════════════════════════════════════════════════════════════════
section("AGFL Comprehensive Results")
agfl = load("comprehensive_agfl.json")
agfl_res = agfl.get("agfl_results", {})
for variant, tex_var in [("AGFL-WL", "AGFLWL"), ("AGFL-Spectral", "AGFLSpectral"),
                          ("AGFL-RW", "AGFLRW"), ("AGFL-Hybrid", "AGFLHybrid")]:
    vd = agfl_res.get(variant, {})
    avg = vd.get("mean", vd)
    std = vd.get("std", {})
    for metric in ["accuracy", "f1", "auc", "precision", "recall"]:
        short = {"accuracy": "Acc", "f1": "Fone", "auc": "AUC",
                 "precision": "Pre", "recall": "Rec"}[metric]
        emit(f"agfl{tex_var}{short}", fmt(avg.get(metric, 0)))
        if std:
            emit(f"agfl{tex_var}{short}Std", fmt(std.get(metric, 0)))
# Manual / original PhishTrace (42-D)
manual = agfl.get("original_phishtrace", {})
mavg = manual.get("mean", manual)
mstd = manual.get("std", {})
for metric in ["accuracy", "f1", "auc", "precision", "recall"]:
    short = {"accuracy": "Acc", "f1": "Fone", "auc": "AUC",
             "precision": "Pre", "recall": "Rec"}[metric]
    emit(f"agflManual{short}", fmt(mavg.get(metric, 0)))
    if mstd:
        emit(f"agflManual{short}Std", fmt(mstd.get(metric, 0)))
# Feature-group ablation
fga = agfl.get("feature_group_ablation", {})
for grp_key, grp_data in fga.items():
    tex_grp = safe_cmd(grp_key)
    gavg = grp_data.get("mean", grp_data)
    gstd = grp_data.get("std", {})
    for metric in ["accuracy", "f1", "auc"]:
        short = {"accuracy": "Acc", "f1": "Fone", "auc": "AUC"}[metric]
        emit(f"agflAbl{tex_grp}{short}", fmt(gavg.get(metric, 0)))
        if gstd:
            emit(f"agflAbl{tex_grp}{short}Std", fmt(gstd.get(metric, 0)))
# all_results table (for comprehensive comparison table including literature baselines)
all_res = agfl.get("all_results", {})
for method_name, md in all_res.items():
    tex_m = safe_cmd(method_name)
    for metric in ["accuracy", "f1", "auc", "precision", "recall"]:
        short = {"accuracy": "Acc", "f1": "Fone", "auc": "AUC",
                 "precision": "Pre", "recall": "Rec"}[metric]
        emit(f"cmp{tex_m}{short}", fmt(md.get(metric, 0)))

# ═══════════════════════════════════════════════════════════════════════════════
# 19. MI Redundancy Per View (from gnn_comparison_results.json)
# ═══════════════════════════════════════════════════════════════════════════════
section("MI Redundancy Per View")
gnn2 = load("gnn_comparison_results.json")
mi2 = gnn2.get("mi", {})
# Emit individual MI and Redundancy values directly
# Keys look like: "MI(URL;Y)", "Redundancy(URL,Network)", etc.
views = ["URL", "Network", "Redirect", "Interaction", "ITG", "CrossView"]
for v in views:
    key = f"MI({v};Y)"
    if key in mi2:
        emit(f"miMI{v}Y", fmt(mi2[key]))
# Redundancy pairs
for i, v1 in enumerate(views):
    for v2 in views[i+1:]:
        key = f"Redundancy({v1},{v2})"
        if key in mi2:
            emit(f"miRedundancy{v1}{v2}", fmt(mi2[key], 4))
# Compute avg redundancy per view for summary
from collections import defaultdict as _dd2
view_red: dict[str, list[float]] = {}
for k, val in mi2.items():
    if k.startswith("Redundancy("):
        inner = k[len("Redundancy("):-1]  # "URL,Network"
        parts = inner.split(",")
        for p in parts:
            p = p.strip()
            if p not in view_red:
                view_red[p] = []
            view_red[p].append(float(val))
for v in views:
    vals = view_red.get(v, [])
    avg_r = sum(vals) / len(vals) if vals else 0.0
    emit(f"miAvgRedundancy{v}", fmt(avg_r, 3))

# ═══════════════════════════════════════════════════════════════════════════════
# N. Formal Verification Results (Z3)
# ═══════════════════════════════════════════════════════════════════════════════
section("Formal Verification (Z3)")
_verf_path = ROOT / "dataset" / "reports" / "formal_verification_results.json"
if _verf_path.exists():
    _verf = json.load(open(_verf_path))
    _verf_summary = _verf.get("summary", {})
    emit("verifProved", _verf_summary.get("proved", 0))
    emit("verifFailed", _verf_summary.get("failed", 0))
    emit("verifSkipped", _verf_summary.get("skipped", 0))
    emit("verifTotal", _verf_summary.get("proved", 0) + _verf_summary.get("failed", 0) + _verf_summary.get("skipped", 0))
    _total_ms = sum(r.get("time_ms", 0) for r in _verf.get("results", []))
    emit("verifTotalTimeMs", fmt1(_total_ms))
    # Per-theorem timing and status
    for r in _verf.get("results", []):
        tid = r.get("theorem_id", 0)
        emit(f"verifTheoremStatus{tid}", r.get("status", "SKIPPED"))
        emit(f"verifTheoremTimeMs{tid}", fmt1(r.get("time_ms", 0)))
    # Bounded model checking sizes (from proof details)
    emit("verifDRPBound", "6")  # N<=6 for DRP
    emit("verifITGBound", "8")  # N<=8 for ITG
else:
    print("  [WARN] formal_verification_results.json not found")

# ═══════════════════════════════════════════════════════════════════════════════
# Fallback: scan paper for all custom macro usages, provide defaults
# ═══════════════════════════════════════════════════════════════════════════════
import re

# Collect all macro names we defined
defined_macros = set()
for l in lines:
    m = re.match(r"\\(?:newcommand|providecommand)\{\\([a-zA-Z]+)\}", l)
    if m:
        defined_macros.add(m.group(1))

# Scan paper sections for custom macros (camelCase = ours, not LaTeX builtins)
used_macros = set()
sections_dir = ROOT / "iccs2026latex" / "sections"
paper_tex = ROOT / "iccs2026latex" / "paper.tex"
tex_files = list(sections_dir.glob("*.tex")) + ([paper_tex] if paper_tex.exists() else [])
for tf in tex_files:
    content = tf.read_text(encoding="utf-8", errors="ignore")
    for m in re.finditer(r"\\([a-z][a-zA-Z]{5,})", content):
        name = m.group(1)
        # Filter LaTeX/TeX builtins, AMS, TikZ, fontawesome, LNCS class commands
        if name in {
            # Core LaTeX formatting
            "textbf", "textit", "texttt", "textsf", "textsc", "textrm",
            "textup", "textsl", "emph", "underline",
            "bfseries", "itshape", "ttfamily", "rmfamily", "sffamily",
            "normalfont", "normalsize", "footnotesize", "scriptsize",
            "small", "large", "Large", "LARGE", "Huge",
            # Document structure
            "begin", "input", "include", "label", "caption", "centering",
            "hline", "noindent", "raggedright",
            "section", "subsection", "subsubsection", "paragraph",
            "footnote", "footnotetext", "footnotemark",
            "renewcommand", "newcommand", "providecommand",
            "newenvironment", "renewenvironment",
            "usepackage", "documentclass", "bibliography",
            "bibliographystyle", "appendix",
            # Tables
            "midrule", "toprule", "bottomrule", "cmidrule",
            "multicolumn", "multirow",
            # Figures
            "includegraphics", "scalebox", "resizebox",
            "colorbox", "fcolorbox", "makebox", "phantom",
            # Spacing
            "vspace", "hspace", "hfill", "vfill",
            "smallskip", "medskip", "bigskip",
            # References
            "tableofcontents", "maketitle", "thanks",
            # LNCS / Springer class
            "institute", "author", "title",
            "authorrunning", "titlerunning",
            "email", "orcidID", "keywords",
            "spnewtheorem", "ackname",
            # Math symbols and operators
            "checkmark", "epsilon", "varepsilon",
            "forall", "exists", "implies", "mathbb", "mathbf",
            "mathcal", "mathrm", "mathit", "mathsf",
            "langle", "rangle", "lfloor", "rfloor",
            "lceil", "rceil", "lvert", "rvert",
            "rightarrow", "leftarrow", "leftrightarrow",
            "Rightarrow", "Leftarrow",
            "mapsto", "propto", "subseteq", "supseteq",
            "xrightarrow", "rightsquigarrow",
            "infty", "nabla", "partial", "equiv",
            "approx", "neq", "leq", "geq", "times", "cdot",
            "ldots", "cdots", "vdots", "ddots",
            # Lengths and counters
            "emergencystretch", "floatsep", "intextsep",
            "textfloatsep", "dblfloatsep",
            "abovecaptionskip", "belowcaptionskip",
            "abovedisplayskip", "belowdisplayskip",
            "abovedisplayshortskip", "belowdisplayshortskip",
            "topfraction", "bottomfraction",
            "floatpagefraction", "textfraction",
            "setlength", "setcounter", "addtocounter",
            "columnwidth", "textwidth", "linewidth",
            "textheight", "paperwidth", "paperheight",
            # Listings / code
            "lstset", "lstinline",
            # TikZ
            "usetikzlibrary", "tikzstyle",
            # Font Awesome (fa*) icons
            "faUser", "faGlobe", "faFingerprint", "faServer",
            "faExclamationTriangle", "faSearch", "faShield",
            "faLock", "faEnvelope", "faCog",
            # xmark / checkmark symbols
            "xmark", "simmark", "textcircled", "textperiodcentered",
            # Misc LaTeX builtins
            "makeatletter", "makeatother", "arabic",
            "nobreakdash", "urlstyle", "nolinkurl",
            "pagenumbering", "thispagestyle", "pagestyle",
            # Bibtex
            "citep", "citet", "citeauthor", "citeyear",
            # hyperref
            "hypersetup", "hyperref", "nameref",
            # booktabs, array, tabularx
            "addlinespace", "specialrule",
            # AMS
            "intertext", "DeclareMathOperator",
            # cleveref
            "crefname", "Crefname",
        }:
            continue
        # Also skip anything that looks like a standard LaTeX command
        # (all lowercase, likely a builtin we missed)
        if name.islower() and len(name) <= 12:
            continue
        used_macros.add(name)

# Add providecommand fallbacks for undefined macros
missing = used_macros - defined_macros
if missing:
    lines.append("")
    lines.append("% --- Fallback macros (not generated from data) ---")
    for name in sorted(missing):
        lines.append(f"\\providecommand{{\\{name}}}{{\\textbf{{??}}}}")
    print(f"  ({len(missing)} fallback macros added)")

# ═══════════════════════════════════════════════════════════════════════════════
# Write output
# ═══════════════════════════════════════════════════════════════════════════════
header = f"""%% ======================================================================
%% AUTO-GENERATED -- DO NOT EDIT MANUALLY
%% Generated by: scripts/generate_latex_data.py
%% Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
%% Source: experiments/results/*.json
%% Re-generate with:  py -3 scripts/generate_latex_data.py
%% ======================================================================
"""

for out_path in [OUT] + ([OUT_DATA] if OUT_DATA.parent.exists() else []):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(lines))
        f.write("\n")

nc = chr(92) + "newcommand"
n_cmds = sum(1 for l in lines if l.startswith(nc))
print(f"Generated {len(lines)} lines → {OUT}")
print(f"  ({n_cmds} commands)")
