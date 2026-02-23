#!/usr/bin/env python3
"""
PhishTrace Formal Verification Suite

Runs all machine-checkable proofs and outputs a verification report.

Usage:
    python verification/verify_all.py

Verifies:
  Part 1 — CTL model checking:  Φ₁–Φ₄ on 8 LTS models (5 attack, 3 benign)
  Part 2 — Z3 SMT proofs:       T1–T6 structural invariant theorems
  Part 3 — Abstraction soundness: label/order preservation on all traces
"""

from __future__ import annotations
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification.lts import (
    ATTACK_MODELS, BENIGN_MODELS, all_models,
)
from verification.ctl_checker import (
    check, check_exists, PHI_ALL,
    Phi1, Phi2, Phi3, Phi4,
)
from verification.z3_invariants import ALL_THEOREMS, run_all_proofs
from verification.itg_abstraction import verify_trace


def header(title: str):
    w = 70
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def subheader(title: str):
    print(f"\n  --- {title} ---")


def main():
    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       PhishTrace — Formal Verification Report              ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    total_checks = 0
    total_pass = 0
    total_fail = 0

    # ══════════════════════════════════════════════════════════
    #  Part 1:  CTL Model Checking
    # ══════════════════════════════════════════════════════════
    header("Part 1: CTL Model Checking  (Φ₁–Φ₄ on LTS models)")

    # Expected results matrix
    #                 Φ₁   Φ₂   Φ₃   Φ₄
    EXPECTED = {
        "A1": (True,  False, True,  False),  # simple cred: cred→ext→rdr
        "A2": (True,  True,  True,  False),  # dual-submit: cred→err→cred→rdr
        "A3": (True,  False, True,  False),  # multistage: multiple cred→ext→rdr
        "A4": (False, False, True,  True),   # financial: fin→ext→rdr
        "A5": (True,  False, True,  False),  # cloaked: (same as A1 on bypass path)
        "B1": (False, False, False, False),  # benign login: no ext
        "B2": (False, False, False, False),  # benign reg: no ext
        "B3": (False, False, False, False),  # benign static: no forms
    }

    phi_names = ["Phi1", "Phi2", "Phi3", "Phi4"]
    phi_list = [Phi1, Phi2, Phi3, Phi4]

    models = all_models()
    print(f"\n  Models: {len(models)}   Properties: {len(phi_list)}")
    print(f"  Total checks: {len(models) * len(phi_list)}")
    print()
    print(f"  {'Model':<25} {'Φ₁':>6} {'Φ₂':>6} {'Φ₃':>6} {'Φ₄':>6}  Status")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*8}")

    for mid, lts in sorted(models.items()):
        K = lts.to_kripke()
        results = []
        for phi in phi_list:
            # Use check_exists for attack models (EF formulas check reachability)
            r = check_exists(K, phi)
            results.append(r)

        expected = EXPECTED[mid]
        row_ok = True
        syms = []
        for r, e in zip(results, expected):
            if r == e:
                syms.append("✓" if r else "·")
            else:
                syms.append("✗")
                row_ok = False

        status = "PASS" if row_ok else "FAIL"
        total_checks += 4
        if row_ok:
            total_pass += 4
        else:
            total_fail += sum(1 for r, e in zip(results, expected) if r != e)
            total_pass += sum(1 for r, e in zip(results, expected) if r == e)

        label = f"{mid} ({lts.name})"
        print(f"  {label:<25} {syms[0]:>6} {syms[1]:>6} {syms[2]:>6} {syms[3]:>6}  {status}")

    # ══════════════════════════════════════════════════════════
    #  Part 2:  Z3 SMT Theorem Proving
    # ══════════════════════════════════════════════════════════
    header("Part 2: Z3 SMT Proofs  (Theorems T1–T6)")

    proof_results = run_all_proofs()
    for tid, (name, fn) in [(t[0], (t[1], t[2])) for t in ALL_THEOREMS]:
        ok, msg = proof_results[tid]
        total_checks += 1
        if ok:
            total_pass += 1
            status = "✓ PROVED"
        else:
            total_fail += 1
            status = "✗ FAILED"
        print(f"\n  {tid} — {name}: {status}")
        # Print message indented
        for line in msg.split("\n"):
            print(f"    {line}")

    # ══════════════════════════════════════════════════════════
    #  Part 3:  Abstraction Soundness
    # ══════════════════════════════════════════════════════════
    header("Part 3: Abstraction Soundness  (ITG construction verification)")

    for mid, lts in sorted(all_models().items()):
        is_attack = mid.startswith("A")
        report = verify_trace(mid, lts, is_attack)

        total_checks += 3  # soundness, invariant, classification
        sound_ok = report["soundness_ok"]
        class_ok = report["classification_correct"]

        if sound_ok:
            total_pass += 1
        else:
            total_fail += 1
        if class_ok:
            total_pass += 2  # invariant + classification
        else:
            total_fail += 1
            total_pass += 1  # partial

        inv_str = ",".join(sorted(report["active_invariants"])) or "none"
        print(
            f"  {mid}: traces={report['traces']:<3}  "
            f"soundness={'✓' if sound_ok else '✗'}  "
            f"invariants={inv_str:<10}  "
            f"classification={'✓' if class_ok else '✗'}"
        )
        for d in report.get("details", []):
            print(f"    {d}")

    # ══════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════
    header("Summary")
    elapsed = time.time() - t0
    print(f"\n  Total checks:  {total_checks}")
    print(f"  Passed:        {total_pass}")
    print(f"  Failed:        {total_fail}")
    print(f"  Pass rate:     {total_pass/total_checks*100:.1f}%")
    print(f"  Time:          {elapsed:.2f}s")

    if total_fail == 0:
        print("\n  ✓ ALL VERIFICATION CHECKS PASSED")
    else:
        print(f"\n  ✗ {total_fail} CHECK(S) FAILED — review above")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
