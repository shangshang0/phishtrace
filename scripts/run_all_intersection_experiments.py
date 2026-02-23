#!/usr/bin/env python3
"""
Run ALL experiments on the intersection dataset (4163 samples).
This ensures every experiment uses the same data for fair comparison.

Runs in order:
1. EITG (main method) - already done, skip if exists
2. GNN comparison
3. Comprehensive + AGFL
4. Depth ablation
5. Adversarial eval
6. Invariance & decoy
7. Trace interpretability
8. Reviewer response experiments
9. Baseline stddevs + Wilcoxon
10. Combined baselines
11. URL adversarial v3
12. Wild detection
13. Feature importance (from EITG)
"""
import json, os, sys, time, traceback
from pathlib import Path
from datetime import datetime

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
sys.path.insert(0, 'experiments')

RESULTS_DIR = Path('experiments/results')
INTERSECT_DIR = Path('experiments/intersection_comparison')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def save_result(name, data):
    """Save to both results/ and intersection_comparison/"""
    # Save to intersection_comparison
    p = INTERSECT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    log(f"  -> Saved {p}")

def run_experiment(name, func, skip_if_exists=False):
    """Run a single experiment with error handling."""
    log(f"\n{'='*60}")
    log(f"RUNNING: {name}")
    log(f"{'='*60}")
    
    out_file = INTERSECT_DIR / f'{name}.json'
    if skip_if_exists and out_file.exists():
        log(f"  Skipping (already exists): {out_file}")
        return json.loads(out_file.read_text(encoding='utf-8'))
    
    t0 = time.time()
    try:
        result = func()
        elapsed = time.time() - t0
        log(f"  Completed in {elapsed:.1f}s")
        if result:
            save_result(name, result)
        return result
    except Exception as e:
        elapsed = time.time() - t0
        log(f"  FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        save_result(name, {'error': str(e), 'traceback': traceback.format_exc()})
        return None

def main():
    INTERSECT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Verify dataset
    from pathlib import Path as P
    traces_dir = P('dataset/traces')
    np = sum(1 for _ in (traces_dir/'phishing').glob('*.json'))
    nb = sum(1 for _ in (traces_dir/'benign').glob('*.json'))
    log(f"Dataset: {np} phishing + {nb} benign = {np+nb} total traces")
    
    all_results = {}
    
    # ── 1. EITG (main method) ──
    def run_eitg():
        from enhanced_itg_detector import run_enhanced_experiments
        run_enhanced_experiments()
        return json.loads((RESULTS_DIR / 'eitg_results.json').read_text(encoding='utf-8'))
    all_results['01_eitg'] = run_experiment('01_eitg_main', run_eitg, skip_if_exists=True)
    
    # ── 2. GNN Comparison ──
    def run_gnn():
        from gnn_comparison import run_gnn_comparison
        return run_gnn_comparison()
    all_results['02_gnn_comparison'] = run_experiment('02_gnn_comparison', run_gnn)
    
    # ── 3. Comprehensive + AGFL ──
    def run_comprehensive():
        from comprehensive_detector import run_comprehensive_experiments
        return run_comprehensive_experiments()
    all_results['03_comprehensive_agfl'] = run_experiment('03_comprehensive_agfl', run_comprehensive)
    
    # ── 4. Depth Ablation ──
    def run_depth():
        from depth_ablation import run_depth_ablation
        return run_depth_ablation()
    all_results['04_depth_ablation'] = run_experiment('04_depth_ablation', run_depth)
    
    # ── 5. Adversarial Eval ──
    def run_adv():
        from adversarial_eval import run_adversarial_evaluation
        return run_adversarial_evaluation()
    all_results['05_adversarial'] = run_experiment('05_adversarial', run_adv)
    
    # ── 6. Invariance & Decoy ──
    def run_inv():
        from invariance_and_decoy_test import main as inv_main
        inv_main()
        return json.loads((RESULTS_DIR / 'invariance_decoy_results.json').read_text(encoding='utf-8'))
    all_results['06_invariance_decoy'] = run_experiment('06_invariance_decoy', run_inv)
    
    # ── 7. Trace Interpretability ──
    def run_interp():
        from trace_interpretability import run_interpretability_demo
        return run_interpretability_demo()
    all_results['07_interpretability'] = run_experiment('07_interpretability', run_interp)
    
    # ── 8. Reviewer Response ──
    def run_reviewer():
        from reviewer_response_experiments import main as rev_main
        rev_main()
        p = RESULTS_DIR / 'reviewer_response_results.json'
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
        return {'status': 'completed_no_file'}
    all_results['08_reviewer_response'] = run_experiment('08_reviewer_response', run_reviewer)
    
    # ── 9. Baseline Stddevs ──
    def run_stddevs():
        from compute_baseline_stddevs import main as std_main
        std_main()
        return json.loads((RESULTS_DIR / 'baseline_stddevs.json').read_text(encoding='utf-8'))
    all_results['09_baseline_stddevs'] = run_experiment('09_baseline_stddevs', run_stddevs)
    
    # ── 10. Combined baselines — REMOVED (run_combined.py deleted) ──
    
    # ── 11. URL Adversarial v3 ──
    def run_url_adv():
        from url_adversarial_eval_v3 import main as uadv_main
        uadv_main()
        p = RESULTS_DIR / 'url_adversarial_v3.json'
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
        return {'status': 'completed'}
    all_results['11_url_adversarial_v3'] = run_experiment('11_url_adversarial_v3', run_url_adv)
    
    # ── 12. Wild Detection — REMOVED (wild_phishing_detection.py deleted) ──
    
    # ── 13. Run original experiments (includes baselines) ──
    def run_original():
        from run_experiments import run_experiments
        our, results = run_experiments()
        return results
    all_results['13_original_baselines'] = run_experiment('13_original_baselines', run_original)
    
    # ── Summary ──
    log(f"\n{'='*70}")
    log("ALL EXPERIMENTS COMPLETE ON INTERSECTION DATASET")
    log(f"{'='*70}")
    
    completed = sum(1 for v in all_results.values() if v is not None)
    failed = sum(1 for v in all_results.values() if v is None)
    log(f"Completed: {completed}, Failed: {failed}")
    
    # Save master summary
    summary = {
        'dataset': {'phishing': np, 'benign': nb, 'total': np + nb},
        'timestamp': datetime.now().isoformat(),
        'experiments_completed': completed,
        'experiments_failed': failed,
        'experiments': {k: ('OK' if v else 'FAILED') for k, v in all_results.items()},
    }
    save_result('00_master_summary', summary)
    
    # List all output files
    log(f"\nOutput files in {INTERSECT_DIR}:")
    for f in sorted(INTERSECT_DIR.glob('*.json')):
        log(f"  {f.name} ({f.stat().st_size:,} bytes)")

if __name__ == '__main__':
    main()
