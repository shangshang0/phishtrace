"""
CTL model checker over finite Kripke structures.

Implements the fixpoint-based algorithm from:
  Baier & Katoen, "Principles of Model Checking", Ch. 6 (MIT Press, 2008).

Supports the full CTL fragment:
  Atom(p), Not(φ), And(φ₁,φ₂), Or(φ₁,φ₂),
  EX(φ), EF(φ), EG(φ), EU(φ₁,φ₂),
  AX(φ), AF(φ), AG(φ), AU(φ₁,φ₂).

Complexity:  O(|φ| · (|S| + |R|))  per formula.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Set

from .lts import KripkeStructure


# ═══════════════════════════════════════════════════════════════
#  CTL syntax
# ═══════════════════════════════════════════════════════════════

class CTL(ABC):
    """Base class for CTL formulas."""

    @abstractmethod
    def __repr__(self) -> str: ...

    def __and__(self, other: "CTL") -> "And":
        return And(self, other)

    def __or__(self, other: "CTL") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True)
class Atom(CTL):
    prop: str
    def __repr__(self) -> str:
        return self.prop


@dataclass(frozen=True)
class TT(CTL):
    """Tautology (true)."""
    def __repr__(self) -> str:
        return "⊤"


@dataclass(frozen=True)
class FF(CTL):
    """Contradiction (false)."""
    def __repr__(self) -> str:
        return "⊥"


@dataclass(frozen=True)
class Not(CTL):
    sub: CTL
    def __repr__(self) -> str:
        return f"¬{self.sub}"


@dataclass(frozen=True)
class And(CTL):
    left: CTL
    right: CTL
    def __repr__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or(CTL):
    left: CTL
    right: CTL
    def __repr__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implies(CTL):
    left: CTL
    right: CTL
    def __repr__(self) -> str:
        return f"({self.left} → {self.right})"


@dataclass(frozen=True)
class EX(CTL):
    sub: CTL
    def __repr__(self) -> str:
        return f"EX({self.sub})"


@dataclass(frozen=True)
class AX(CTL):
    sub: CTL
    def __repr__(self) -> str:
        return f"AX({self.sub})"


@dataclass(frozen=True)
class EF(CTL):
    sub: CTL
    def __repr__(self) -> str:
        return f"EF({self.sub})"


@dataclass(frozen=True)
class AF(CTL):
    sub: CTL
    def __repr__(self) -> str:
        return f"AF({self.sub})"


@dataclass(frozen=True)
class EG(CTL):
    sub: CTL
    def __repr__(self) -> str:
        return f"EG({self.sub})"


@dataclass(frozen=True)
class AG(CTL):
    sub: CTL
    def __repr__(self) -> str:
        return f"AG({self.sub})"


@dataclass(frozen=True)
class EU(CTL):
    left: CTL
    right: CTL
    def __repr__(self) -> str:
        return f"E[{self.left} U {self.right}]"


@dataclass(frozen=True)
class AU(CTL):
    left: CTL
    right: CTL
    def __repr__(self) -> str:
        return f"A[{self.left} U {self.right}]"


# ═══════════════════════════════════════════════════════════════
#  Model checker — SAT : K × CTL → 2^S
# ═══════════════════════════════════════════════════════════════

def sat(K: KripkeStructure, phi: CTL) -> Set[int]:
    """
    Return the set of states in K that satisfy φ.

    Uses the standard fixpoint characterisation:
      SAT(EG φ) = νX. SAT(φ) ∩ pre∃(X)
      SAT(E[φ₁ U φ₂]) = μX. SAT(φ₂) ∪ (SAT(φ₁) ∩ pre∃(X))
    """
    S = K.states

    if isinstance(phi, TT):
        return set(S)

    if isinstance(phi, FF):
        return set()

    if isinstance(phi, Atom):
        return {s for s in S if phi.prop in K.L.get(s, set())}

    if isinstance(phi, Not):
        return S - sat(K, phi.sub)

    if isinstance(phi, And):
        return sat(K, phi.left) & sat(K, phi.right)

    if isinstance(phi, Or):
        return sat(K, phi.left) | sat(K, phi.right)

    if isinstance(phi, Implies):
        # p → q  ≡  ¬p ∨ q
        return sat(K, Or(Not(phi.left), phi.right))

    if isinstance(phi, EX):
        sub = sat(K, phi.sub)
        return _pre_exists(K, sub)

    if isinstance(phi, AX):
        # AX φ ≡ ¬EX(¬φ)
        return S - _pre_exists(K, S - sat(K, phi.sub))

    if isinstance(phi, EF):
        # EF φ ≡ E[⊤ U φ]
        return sat(K, EU(TT(), phi.sub))

    if isinstance(phi, AF):
        # AF φ ≡ ¬EG(¬φ)
        return S - sat(K, EG(Not(phi.sub)))

    if isinstance(phi, EG):
        # Greatest fixpoint: νX. SAT(φ) ∩ pre∃(X)
        sub = sat(K, phi.sub)
        X = set(sub)
        while True:
            X_new = sub & _pre_exists(K, X)
            if X_new == X:
                return X
            X = X_new

    if isinstance(phi, AG):
        # AG φ ≡ ¬EF(¬φ)
        return S - sat(K, EF(Not(phi.sub)))

    if isinstance(phi, EU):
        # Least fixpoint: μX. SAT(φ₂) ∪ (SAT(φ₁) ∩ pre∃(X))
        s1 = sat(K, phi.left)
        s2 = sat(K, phi.right)
        X = set(s2)
        while True:
            X_new = s2 | (s1 & _pre_exists(K, X))
            if X_new == X:
                return X
            X = X_new

    if isinstance(phi, AU):
        # A[φ₁ U φ₂] ≡ ¬(E[¬φ₂ U (¬φ₁ ∧ ¬φ₂)] ∨ EG(¬φ₂))
        not1 = Not(phi.left)
        not2 = Not(phi.right)
        return S - (sat(K, EU(not2, And(not1, not2))) | sat(K, EG(not2)))

    raise TypeError(f"Unknown CTL connective: {type(phi).__name__}")


def _pre_exists(K: KripkeStructure, target: Set[int]) -> Set[int]:
    """pre∃(T) = { s ∈ S | ∃s' ∈ T. (s, s') ∈ R }"""
    return {s for s in range(K.n) if K.R[s] & target}


# ── convenience ────────────────────────────────────────────────

def check(K: KripkeStructure, phi: CTL) -> bool:
    """Does φ hold in all initial states of K?"""
    return K.initial <= sat(K, phi)


def check_exists(K: KripkeStructure, phi: CTL) -> bool:
    """Does φ hold in at least one initial state of K?"""
    return bool(K.initial & sat(K, phi))


# ═══════════════════════════════════════════════════════════════
#  PhishTrace detection properties  Φ₁ – Φ₄
# ═══════════════════════════════════════════════════════════════

# Atomic propositions
cred  = Atom("cred")
ext   = Atom("ext")
err   = Atom("err")
rdr   = Atom("rdr")
sens  = Atom("sens")
fin   = Atom("fin")
legit = Atom("legit")

# ── Φ₁  Credential exfiltration ──
# "There exists a path reaching a credential-input state
#  from which an external-communication state is reachable."
Phi1 = EF(cred & EF(ext))

# ── Φ₂  Dual-submission ──
# "There exists a path: cred → … → err → … → cred → … → rdr"
Phi2 = EF(cred & EF(err & EF(cred & EF(rdr))))

# ── Φ₃  Redirect-chain anomaly ──
# "An external-communication state is reachable and followed by a redirect."
Phi3 = EF(ext & EF(rdr))

# ── Φ₄  Financial exfiltration ──
# "Financial-data state exists and leads to external communication."
Phi4 = EF(fin & EF(ext))

PHI_ALL = {"Phi1": Phi1, "Phi2": Phi2, "Phi3": Phi3, "Phi4": Phi4}
