"""
Formal verification package for PhishTrace.

Provides machine-checkable proofs that the detection method is
sound and complete for the defined attack class. Consists of:

  lts.py           — Labeled Transition System + Kripke structures
  ctl_checker.py   — CTL model checker over finite Kripke structures
  z3_invariants.py — Z3 SMT proofs of structural invariants
  verify_all.py    — Run all verification checks
"""
