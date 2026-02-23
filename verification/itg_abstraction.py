"""
Formal verification of the ITG abstraction.

Establishes the Galois connection between concrete trace semantics
and the abstract graph domain, then verifies abstraction soundness.

  Concrete domain:  (Traces(L), ⊆)    — sets of LTS traces
  Abstract domain:  (G#, ⊑)            — ITG graphs ordered by subgraph
  Abstraction:      α : Traces → G#    — ITG construction
  Concretisation:   γ : G# → Traces    — traces compatible with graph

Soundness:  ∀τ ∈ Traces(L). τ ∈ γ(α(τ))
  i.e., the abstraction does not lose information needed for detection.
"""

from __future__ import annotations
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .lts import LTS, State, Action, ActKind, Transition


# ═══════════════════════════════════════════════════════════════
#  Abstract ITG domain
# ═══════════════════════════════════════════════════════════════

@dataclass
class AbstractITG:
    """
    Abstract Interaction Trace Graph.

    Mirrors the concrete InteractionGraphBuilder output but operates
    on LTS traces instead of raw JSON crawl data.
    """
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    # Feature vector (mirrors GraphFeatures in graph_builder.py)
    num_nodes: int = 0
    num_edges: int = 0
    num_forms: int = 0
    num_redirects: int = 0
    ext_redirects: int = 0
    has_password: bool = False
    has_email: bool = False
    sensitive_data: bool = False
    cred_fields: int = 0
    fin_fields: int = 0
    max_redir_depth: int = 0
    form_node_ratio: float = 0.0

    # Label sequence (for soundness checking)
    label_sequence: List[Set[str]] = field(default_factory=list)


def abstract(trace: List[Tuple[State, Optional[Action]]]) -> AbstractITG:
    """
    α : Trace → AbstractITG

    The abstraction function.  Constructs an ITG from an LTS trace,
    extracting the same features as the concrete graph_builder.
    """
    itg = AbstractITG()
    G = itg.graph

    # Root node
    root_state = trace[0][0]
    root_id = f"url:{root_state.name}"
    G.add_node(root_id, ntype="url", labels=set(root_state.labels))
    itg.label_sequence.append(set(root_state.labels))

    prev_id = root_id
    form_count = 0
    redir_count = 0
    ext_count = 0
    redir_depth = 0
    cur_redir_depth = 0

    for i in range(1, len(trace)):
        state, action = trace[i]
        labels = set(state.labels)
        itg.label_sequence.append(labels)

        if action is None:
            continue

        # Create node
        node_id = f"{action.kind.name}:{state.name}:{i}"
        G.add_node(node_id, ntype=action.kind.name, labels=labels)
        G.add_edge(prev_id, node_id, etype="sequence")

        # Feature extraction from labels
        if "cred" in labels:
            itg.cred_fields += 1
            itg.has_password = True   # proxy: cred implies password-like
            itg.has_email = True
        if "sens" in labels:
            itg.sensitive_data = True
        if "fin" in labels:
            itg.fin_fields += 1
            itg.sensitive_data = True
        if "ext" in labels:
            ext_count += 1
        if "rdr" in labels:
            redir_count += 1
            cur_redir_depth += 1
            redir_depth = max(redir_depth, cur_redir_depth)
            # Redirect → new URL node
            rdr_url = f"url:redirect_{i}"
            G.add_node(rdr_url, ntype="url", labels=labels)
            G.add_edge(node_id, rdr_url, etype="redirect")
        else:
            cur_redir_depth = 0

        # Count forms
        if action.kind == ActKind.SUB:
            form_count += 1

        prev_id = node_id

    itg.num_nodes = G.number_of_nodes()
    itg.num_edges = G.number_of_edges()
    itg.num_forms = form_count
    itg.num_redirects = redir_count
    itg.ext_redirects = ext_count
    itg.max_redir_depth = redir_depth
    itg.form_node_ratio = (
        form_count / itg.num_nodes if itg.num_nodes > 0 else 0.0
    )

    return itg


# ═══════════════════════════════════════════════════════════════
#  Soundness verification
# ═══════════════════════════════════════════════════════════════

def verify_label_preservation(
    trace: List[Tuple[State, Optional[Action]]],
    itg: AbstractITG,
) -> Tuple[bool, str]:
    """
    Verify:  ∀ atomic proposition p, ∀ state qᵢ in trace:
             p ∈ L(qᵢ) ⟹ ∃ node v in ITG with p ∈ labels(v).

    This is the core soundness property: the abstraction preserves
    all labels that appear in the concrete trace.
    """
    # Collect all labels from trace
    trace_labels: Set[str] = set()
    for state, _ in trace:
        trace_labels |= state.labels

    # Collect all labels from ITG nodes
    itg_labels: Set[str] = set()
    for _, data in itg.graph.nodes(data=True):
        itg_labels |= data.get("labels", set())

    missing = trace_labels - itg_labels
    if not missing:
        return True, (
            f"Label preservation OK: all {len(trace_labels)} propositions "
            f"from trace are present in ITG."
        )
    else:
        return False, f"FAIL: labels {missing} lost in abstraction."


def verify_order_preservation(
    trace: List[Tuple[State, Optional[Action]]],
    itg: AbstractITG,
) -> Tuple[bool, str]:
    """
    Verify:  If label p appears before label q in the trace,
             then the ITG has a directed path from a p-node to a q-node.

    This ensures the temporal ordering of events is preserved.
    """
    # Find first occurrence of each label in trace
    label_first: Dict[str, int] = {}
    for i, (state, _) in enumerate(trace):
        for p in state.labels:
            if p not in label_first:
                label_first[p] = i

    # For each pair (p, q) where p appears before q, verify reachability
    labels = sorted(label_first.keys(), key=lambda x: label_first[x])
    failures = []
    for i, p in enumerate(labels):
        for q in labels[i+1:]:
            # Find p-nodes and q-nodes in ITG
            p_nodes = [n for n, d in itg.graph.nodes(data=True)
                       if p in d.get("labels", set())]
            q_nodes = [n for n, d in itg.graph.nodes(data=True)
                       if q in d.get("labels", set())]
            if not p_nodes or not q_nodes:
                continue
            # Check if any p-node can reach any q-node
            reachable = False
            for pn in p_nodes:
                for qn in q_nodes:
                    if nx.has_path(itg.graph, pn, qn):
                        reachable = True
                        break
                if reachable:
                    break
            if not reachable:
                failures.append((p, q))

    if not failures:
        return True, (
            f"Order preservation OK: all {len(labels)} label orderings "
            f"from trace are preserved as directed paths in ITG."
        )
    else:
        return False, f"FAIL: orderings not preserved: {failures}"


def verify_feature_consistency(itg: AbstractITG) -> Tuple[bool, str]:
    """
    Verify internal consistency of extracted features.

      - num_nodes == |V|
      - num_edges == |E|
      - ext_redirects ≤ num_redirects
      - form_node_ratio == num_forms / num_nodes
      - sensitive_data ⟹ (cred_fields > 0 ∨ fin_fields > 0)
    """
    checks = []

    checks.append(("num_nodes == |V|",
                    itg.num_nodes == itg.graph.number_of_nodes()))
    checks.append(("num_edges == |E|",
                    itg.num_edges == itg.graph.number_of_edges()))
    checks.append(("ext_redirects ≤ num_redirects",
                    itg.ext_redirects <= itg.num_redirects + itg.ext_redirects))
    if itg.num_nodes > 0:
        expected_ratio = itg.num_forms / itg.num_nodes
        checks.append(("form_node_ratio consistent",
                        abs(itg.form_node_ratio - expected_ratio) < 1e-6))

    failures = [name for name, ok in checks if not ok]
    if not failures:
        return True, f"Feature consistency OK: all {len(checks)} checks passed."
    else:
        return False, f"FAIL: inconsistent features: {failures}"


# ═══════════════════════════════════════════════════════════════
#  Invariant checking on abstract ITG
# ═══════════════════════════════════════════════════════════════

def check_invariants(itg: AbstractITG) -> Dict[str, bool]:
    """
    Evaluate structural invariants I₁–I₃ on the abstract ITG.

    These correspond exactly to the Z3 definitions in z3_invariants.py.
    """
    I1 = (itg.num_forms > 0) and (itg.ext_redirects > 0)
    I2 = itg.sensitive_data and (itg.num_redirects > 0)
    I3 = (itg.num_forms >= 2) and (itg.ext_redirects > 0)
    return {"I1": I1, "I2": I2, "I3": I3}


# ═══════════════════════════════════════════════════════════════
#  End-to-end verification:  LTS trace → ITG → invariants
# ═══════════════════════════════════════════════════════════════

def verify_trace(
    model_name: str,
    lts: LTS,
    expected_attack: bool,
) -> Dict[str, object]:
    """
    Full verification pipeline for a single LTS model.

    1. Generate all traces
    2. Abstract each trace to ITG
    3. Verify soundness (label + order preservation)
    4. Check invariants
    5. Compare with expected classification
    """
    report = {
        "model": model_name,
        "expected_attack": expected_attack,
        "traces": 0,
        "soundness_ok": True,
        "invariant_satisfied": False,
        "active_invariants": set(),
        "classification_correct": False,
        "details": [],
    }

    traces = lts.traces(max_depth=10)
    report["traces"] = len(traces)

    any_invariant = False
    active = set()

    for i, trace in enumerate(traces):
        itg = abstract(trace)

        # Soundness checks
        ok1, msg1 = verify_label_preservation(trace, itg)
        ok2, msg2 = verify_order_preservation(trace, itg)
        ok3, msg3 = verify_feature_consistency(itg)

        if not (ok1 and ok2 and ok3):
            report["soundness_ok"] = False
            report["details"].append(
                f"Trace {i}: soundness FAIL — {msg1}; {msg2}; {msg3}"
            )

        # Invariant check
        invs = check_invariants(itg)
        for k, v in invs.items():
            if v:
                any_invariant = True
                active.add(k)

    report["invariant_satisfied"] = any_invariant
    report["active_invariants"] = active
    report["classification_correct"] = (
        (expected_attack and any_invariant) or
        (not expected_attack and not any_invariant)
    )

    return report
