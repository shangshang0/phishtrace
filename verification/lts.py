"""
Labeled Transition System (LTS) and Kripke structures for
modelling web interaction systems.

Notation follows Baier & Katoen, "Principles of Model Checking" (MIT Press, 2008).

  L  = (Q, q0, Act, -->, AP, L)
  Q  : finite set of states
  q0 : initial state
  Act: finite set of actions, partitioned into Inp | Nav | Sys
  -->: Q x Act x Q  transition relation
  AP : {cred, ext, err, rdr, sens, fin, legit}
  L  : Q -> 2^AP   labelling function
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, FrozenSet, List, Optional, Set, Tuple,
)


# ── Action taxonomy ──────────────────────────────────────────────

class ActKind(Enum):
    """Partition of the action alphabet."""
    INP = auto()   # user input  (fill, type)
    SUB = auto()   # form submit
    CLK = auto()   # click
    NAV = auto()   # page navigation
    RDR = auto()   # server-side redirect
    SYS = auto()   # load / error / system response


# ── Atomic propositions ─────────────────────────────────────────

AP_SET = frozenset({"cred", "ext", "err", "rdr", "sens", "fin", "legit"})


# ── Core data types ─────────────────────────────────────────────

@dataclass(frozen=True)
class State:
    """A state in the LTS, labelled with atomic propositions."""
    name: str
    labels: FrozenSet[str] = field(default_factory=frozenset)

    def has(self, p: str) -> bool:
        return p in self.labels

    def __repr__(self) -> str:
        lbl = ",".join(sorted(self.labels)) if self.labels else "∅"
        return f"{self.name}[{lbl}]"


@dataclass(frozen=True)
class Action:
    name: str
    kind: ActKind

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Transition:
    src: State
    act: Action
    tgt: State


# ── LTS ─────────────────────────────────────────────────────────

class LTS:
    """Labeled Transition System."""

    def __init__(self, name: str = ""):
        self.name = name
        self.states: Set[State] = set()
        self.initial: Optional[State] = None
        self.actions: Set[Action] = set()
        self.trans: Set[Transition] = set()

    # -- construction helpers --

    def state(self, name: str, labels: Set[str] | None = None) -> State:
        s = State(name, frozenset(labels or set()))
        self.states.add(s)
        if self.initial is None:
            self.initial = s
        return s

    def edge(self, src: State, act_name: str, kind: ActKind, tgt: State) -> Transition:
        a = Action(act_name, kind)
        self.actions.add(a)
        t = Transition(src, a, tgt)
        self.trans.add(t)
        return t

    # -- queries --

    def post(self, s: State) -> List[Tuple[Action, State]]:
        """Successor states with their actions."""
        return [(t.act, t.tgt) for t in self.trans if t.src == s]

    def reachable(self, depth: int = 50) -> Set[State]:
        """BFS reachability from initial state."""
        visited: Set[State] = set()
        frontier = {self.initial}
        d = 0
        while frontier and d < depth:
            visited |= frontier
            nxt: Set[State] = set()
            for s in frontier:
                for _, t in self.post(s):
                    if t not in visited:
                        nxt.add(t)
            frontier = nxt
            d += 1
        return visited

    def traces(self, max_depth: int = 8) -> List[List[Tuple[State, Optional[Action]]]]:
        """Enumerate all acyclic traces up to max_depth."""
        results: List[List[Tuple[State, Optional[Action]]]] = []
        stack: List[Tuple[List[Tuple[State, Optional[Action]]], Set[State]]] = [
            ([(self.initial, None)], {self.initial})
        ]
        while stack:
            path, visited = stack.pop()
            cur = path[-1][0]
            extended = False
            if len(path) <= max_depth:
                for act, tgt in self.post(cur):
                    if tgt not in visited:
                        extended = True
                        stack.append(
                            (path + [(tgt, act)], visited | {tgt})
                        )
            if not extended:
                results.append(path)
        return results

    # -- conversion --

    def to_kripke(self) -> "KripkeStructure":
        """Convert to Kripke structure for model checking."""
        sl = sorted(self.states, key=lambda s: s.name)
        idx = {s: i for i, s in enumerate(sl)}
        n = len(sl)
        R: List[Set[int]] = [set() for _ in range(n)]
        for t in self.trans:
            R[idx[t.src]].add(idx[t.tgt])
        # make total: if no successor, add self-loop (standard trick)
        for i in range(n):
            if not R[i]:
                R[i].add(i)
        L = {i: set(sl[i].labels) for i in range(n)}
        return KripkeStructure(
            n=n,
            initial={idx[self.initial]},
            R=R,
            L=L,
            names={i: sl[i].name for i in range(n)},
        )


# ── Kripke Structure ────────────────────────────────────────────

@dataclass
class KripkeStructure:
    """
    Kripke structure  K = (S, S0, R, L)  over AP.

    Representation:
      n       — number of states  (states are 0…n-1)
      initial — set of initial states
      R       — R[i] is the set of successors of state i  (total)
      L       — L[i] is the set of atomic propositions true at state i
      names   — human-readable state names
    """
    n: int
    initial: Set[int]
    R: List[Set[int]]
    L: Dict[int, Set[str]]
    names: Dict[int, str] = field(default_factory=dict)

    @property
    def states(self) -> Set[int]:
        return set(range(self.n))


# ═══════════════════════════════════════════════════════════════
#  Pre-built attack and benign models
# ═══════════════════════════════════════════════════════════════

def make_simple_credential_harvest() -> LTS:
    """
    Attack A1 — Simple credential harvest.

    Path:  landing ─[fill]→ form ─[submit]→ exfil ─[rdr]→ done
    Labels: form={cred,sens}, exfil={ext}, done={rdr}
    """
    m = LTS("A1-simple-cred")
    q0 = m.state("landing", set())
    q1 = m.state("form", {"cred", "sens"})
    q2 = m.state("exfil", {"ext"})
    q3 = m.state("done", {"rdr"})
    m.edge(q0, "load", ActKind.SYS, q1)
    m.edge(q1, "fill_cred", ActKind.INP, q1)
    m.edge(q1, "submit", ActKind.SUB, q2)
    m.edge(q2, "redirect", ActKind.RDR, q3)
    return m


def make_dual_submission() -> LTS:
    """
    Attack A2 — Dual-submission attack.

    Path:  landing → form → submit → error → re-fill → submit₂ → legit
    The error page re-shows the form, victim re-enters credentials.
    """
    m = LTS("A2-dual-submit")
    q0 = m.state("landing", set())
    q1 = m.state("form1", {"cred", "sens"})
    q2 = m.state("submit1", {"ext"})
    q3 = m.state("error", {"err", "cred", "sens"})
    q4 = m.state("submit2", {"ext"})
    q5 = m.state("redirect", {"rdr", "legit"})
    m.edge(q0, "load", ActKind.SYS, q1)
    m.edge(q1, "fill_cred", ActKind.INP, q1)
    m.edge(q1, "submit", ActKind.SUB, q2)
    m.edge(q2, "show_error", ActKind.SYS, q3)
    m.edge(q3, "re_fill", ActKind.INP, q3)
    m.edge(q3, "submit", ActKind.SUB, q4)
    m.edge(q4, "redirect", ActKind.RDR, q5)
    return m


def make_multistage() -> LTS:
    """
    Attack A3 — Multi-stage collection.

    Stage 1: email → Stage 2: password → Stage 3: OTP → exfil
    """
    m = LTS("A3-multistage")
    q0 = m.state("landing", set())
    q1 = m.state("email_form", {"cred"})
    q2 = m.state("password_form", {"cred", "sens"})
    q3 = m.state("otp_form", {"cred", "sens"})
    q4 = m.state("exfil", {"ext"})
    q5 = m.state("done", {"rdr", "legit"})
    m.edge(q0, "load", ActKind.SYS, q1)
    m.edge(q1, "fill_email", ActKind.INP, q1)
    m.edge(q1, "next", ActKind.SUB, q2)
    m.edge(q2, "fill_pass", ActKind.INP, q2)
    m.edge(q2, "next", ActKind.SUB, q3)
    m.edge(q3, "fill_otp", ActKind.INP, q3)
    m.edge(q3, "submit", ActKind.SUB, q4)
    m.edge(q4, "redirect", ActKind.RDR, q5)
    return m


def make_financial_phish() -> LTS:
    """
    Attack A4 — Financial data phishing.

    Collects credit card / bank info then redirects.
    """
    m = LTS("A4-financial")
    q0 = m.state("landing", set())
    q1 = m.state("card_form", {"fin", "sens"})
    q2 = m.state("exfil", {"ext"})
    q3 = m.state("done", {"rdr"})
    m.edge(q0, "load", ActKind.SYS, q1)
    m.edge(q1, "fill_card", ActKind.INP, q1)
    m.edge(q1, "submit", ActKind.SUB, q2)
    m.edge(q2, "redirect", ActKind.RDR, q3)
    return m


def make_cloaked_phish() -> LTS:
    """
    Attack A5 — Cloaked phishing (anti-bot awareness).

    Landing page checks UA; serves benign if bot detected,
    phishing otherwise.  The deep crawler bypasses cloaking.
    """
    m = LTS("A5-cloaked")
    q0 = m.state("landing", set())
    # bot path
    q_benign = m.state("decoy", {"legit"})
    # real user / bypassed path
    q1 = m.state("real_form", {"cred", "sens"})
    q2 = m.state("exfil", {"ext"})
    q3 = m.state("done", {"rdr"})
    m.edge(q0, "detect_bot", ActKind.SYS, q_benign)
    m.edge(q0, "bypass", ActKind.SYS, q1)
    m.edge(q1, "fill_cred", ActKind.INP, q1)
    m.edge(q1, "submit", ActKind.SUB, q2)
    m.edge(q2, "redirect", ActKind.RDR, q3)
    return m


def make_benign_login() -> LTS:
    """
    Benign B1 — Legitimate login page.

    login → submit → dashboard (same domain, no ext)
    """
    m = LTS("B1-benign-login")
    q0 = m.state("landing", {"legit"})
    q1 = m.state("login_form", {"cred", "legit"})
    q2 = m.state("dashboard", {"legit"})
    m.edge(q0, "load", ActKind.SYS, q1)
    m.edge(q1, "fill_cred", ActKind.INP, q1)
    m.edge(q1, "submit", ActKind.SUB, q2)
    return m


def make_benign_registration() -> LTS:
    """
    Benign B2 — Legitimate multi-step registration.

    Collects email → password → profile → confirmation.
    All on same domain, no external exfiltration.
    """
    m = LTS("B2-benign-reg")
    q0 = m.state("landing", {"legit"})
    q1 = m.state("email_step", {"cred", "legit"})
    q2 = m.state("pass_step", {"cred", "sens", "legit"})
    q3 = m.state("profile_step", {"legit"})
    q4 = m.state("confirm", {"legit"})
    m.edge(q0, "load", ActKind.SYS, q1)
    m.edge(q1, "fill_email", ActKind.INP, q1)
    m.edge(q1, "next", ActKind.SUB, q2)
    m.edge(q2, "fill_pass", ActKind.INP, q2)
    m.edge(q2, "next", ActKind.SUB, q3)
    m.edge(q3, "fill_profile", ActKind.INP, q3)
    m.edge(q3, "submit", ActKind.SUB, q4)
    return m


def make_benign_static() -> LTS:
    """
    Benign B3 — Static information page (no forms).
    """
    m = LTS("B3-benign-static")
    q0 = m.state("page", {"legit"})
    q1 = m.state("about", {"legit"})
    q2 = m.state("contact", {"legit"})
    m.edge(q0, "click_about", ActKind.CLK, q1)
    m.edge(q0, "click_contact", ActKind.CLK, q2)
    m.edge(q1, "back", ActKind.NAV, q0)
    m.edge(q2, "back", ActKind.NAV, q0)
    return m


# ── Catalogue ───────────────────────────────────────────────────

ATTACK_MODELS = {
    "A1": make_simple_credential_harvest,
    "A2": make_dual_submission,
    "A3": make_multistage,
    "A4": make_financial_phish,
    "A5": make_cloaked_phish,
}

BENIGN_MODELS = {
    "B1": make_benign_login,
    "B2": make_benign_registration,
    "B3": make_benign_static,
}


def all_models() -> Dict[str, LTS]:
    d: Dict[str, LTS] = {}
    for k, f in {**ATTACK_MODELS, **BENIGN_MODELS}.items():
        d[k] = f()
    return d
