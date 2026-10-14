# Author: abdkhan
"""Neuro-symbolic verifier: gates every tiny-LLM-proposed intervention
against the current causal graph before it is allowed to act on the
IoT edge node. This is the paper's answer to "LLMs are not calibrated,
reliable causal reasoners" (arXiv:2608.23660) -- the tiny LLM is fast but
unverified; the graph is slow-to-update but symbolically trustworthy.

Failure modes it is designed to catch:
  1. UNPARSEABLE   -- the LLM didn't emit a well-formed do(var=value) action.
  2. NON_ACTIONABLE -- the LLM tried to "intervene" on a sensed/outcome
     variable that a real IoT node cannot directly set (e.g. do(latency=...)).
  3. OUT_OF_RANGE / RATE_LIMIT -- value outside [0,1] or exceeds the max
     physically plausible step change for one control tick.
  4. SPURIOUS_EFFECT -- the LLM claims a directional effect on a QoS variable
     that has no directed path from the intervened variable in the current
     discovered graph (a hallucinated causal claim).
  5. SIGN_MISMATCH -- there IS a path, but the *sign* of the claimed effect
     disagrees with the sign implied by a linear refit of the discovered
     edges (e.g. LLM says "duty_cycle up -> latency down", graph says up).
"""
from __future__ import annotations

import re
import dataclasses
from typing import Optional

import numpy as np
import networkx as nx
import pandas as pd

ACTIONABLE_VARS = {"duty_cycle", "tx_power"}
QOS_VARS = ["latency", "throughput", "packet_loss", "energy"]
MAX_STEP = 0.35  # actuator rate limit per control tick
NO_OP_EPS = 0.01  # below this the intervention cannot move the plant at all

# Matches a do(...) call and captures its whole argument list, so that a
# MULTI-variable call -- "do(duty_cycle=0.2, transmit_power=0.75)" -- is read
# rather than discarded. A single-pair regex ending in `\)` silently failed on
# those, and a degraded model emits them often enough that they accounted for
# 95% of one condition's rejections; scoring a parser limitation as if it were
# an attack signature would have been a reporting error, not a finding.
DO_PATTERN = re.compile(r"do\(\s*([^()]*?)\s*\)")
_DO_ARG = re.compile(r"([a-zA-Z_]+)\s*=\s*(-?[0-9]*\.?[0-9]+)")
# The gap between the QoS variable and its direction word must not run past
# a clause boundary (';') or a sentence end ('. ' / '.<end>'), but it MUST be
# allowed to contain decimal points: the proposal-style outputs the control
# loop actually produces read
#     "packet loss rate moves from 0.37 to 0.31 (decrease by about 6.0 ...)"
# and an earlier `[^.;]*?` gap silently failed on the "0.37"/"0.31" decimals,
# so parse_claims returned [] for every proposal-style output and the
# sign-consistency check was inert in exactly the closed-loop experiments it
# mattered most for. Bounded to 80 chars so a claim cannot bind to a
# direction word belonging to a different variable further along.
EFFECT_PATTERN = re.compile(
    r"(end-to-end latency|energy consumption|latency|throughput|packet[_ ]loss|energy)"
    r"(?:(?!;)(?!\.\s)(?!\.$).){0,80}?\b(increase|decrease|rise|drop|reduce|grow)\b",
    re.IGNORECASE)

_EFFECT_VAR_MAP = {
    "latency": "latency", "end-to-end latency": "latency",
    "throughput": "throughput",
    "packet_loss": "packet_loss", "packet loss": "packet_loss",
    "energy": "energy", "energy consumption": "energy",
}
_DECREASE_WORDS = {"decrease", "drop", "reduce"}


@dataclasses.dataclass
class VerifierResult:
    verdict: str          # ACCEPT | CLIP | REJECT
    reason: str
    safe_action: dict     # {"duty_cycle": x, "tx_power": y} to actually execute
    parsed_do: Optional[dict] = None
    flagged_claims: Optional[list] = None


def parse_action(text: str) -> Optional[dict]:
    """Extract the intervention from a do(...) call.

    Returns a single {var: value}. When the model emits several arguments in
    one call, the first ACTIONABLE variable wins (a controller can only move
    an actuator; a do() naming an outcome variable is not an intervention it
    can execute). Falling back to the first argument otherwise preserves the
    NON_ACTIONABLE rejection path for calls like do(latency=0.5).
    """
    for m in DO_PATTERN.finditer(text):
        args = _DO_ARG.findall(m.group(1))
        if not args:
            continue
        for var, val in args:
            if var in ACTIONABLE_VARS:
                return {var: float(val)}
        var, val = args[0]
        return {var: float(val)}
    return None


def parse_claims(text: str) -> list[tuple[str, int]]:
    """Return list of (qos_var, sign) claimed in free text, sign=+1 increase, -1 decrease."""
    claims = []
    for m in EFFECT_PATTERN.finditer(text):
        raw_var = m.group(1).lower()
        var = _EFFECT_VAR_MAP.get(raw_var)
        if var is None:
            continue
        sign = -1 if m.group(2).lower() in _DECREASE_WORDS else 1
        claims.append((var, sign))
    return claims


def fit_edge_signs(df: pd.DataFrame, graph: nx.DiGraph) -> dict:
    """Cheap per-edge sign estimate: correlation sign of (parent, child) in
    the observational data, used to sanity check the graph-implied direction
    of an intervention's effect without needing a full linear SEM refit."""
    signs = {}
    for u, v in graph.edges():
        if u in df.columns and v in df.columns:
            c = np.corrcoef(df[u].to_numpy(), df[v].to_numpy())[0, 1]
            signs[(u, v)] = 1 if c >= 0 else -1
    return signs


def implied_sign(graph: nx.DiGraph, edge_signs: dict, src: str, dst: str,
                  max_paths: int = 8, cutoff: int = 6) -> Optional[int]:
    """Sign of the total causal effect of src on dst, aggregated by majority
    vote of the per-path sign (product of edge signs) over all simple
    directed paths from src to dst (capped for tractability). Using every
    path rather than a single shortest path avoids a fragile verdict that
    hinges on whichever one edge a shortest-path search happens to pick --
    a single poisoned edge should not flip the verdict if other, unpoisoned
    routes still agree on the true direction of effect.
    Returns None (spurious / no confident path) if there is no path, or if
    votes are exactly tied.
    """
    if src not in graph or dst not in graph or not nx.has_path(graph, src, dst):
        return None
    votes = []
    for i, path in enumerate(nx.all_simple_paths(graph, src, dst, cutoff=cutoff)):
        if i >= max_paths:
            break
        sign = 1
        ok = True
        for a, b in zip(path[:-1], path[1:]):
            s = edge_signs.get((a, b))
            if s is None:
                ok = False
                break
            sign *= s
        if ok:
            votes.append(sign)
    if not votes:
        return None
    total = sum(votes)
    if total == 0:
        return None
    return 1 if total > 0 else -1


class CausalVerifier:
    def __init__(self, graph: nx.DiGraph, obs_df: pd.DataFrame):
        self.graph = graph
        self.edge_signs = fit_edge_signs(obs_df, graph)

    def verify(self, llm_output: str, current_action: dict) -> VerifierResult:
        parsed = parse_action(llm_output)
        if parsed is None:
            return VerifierResult("REJECT", "UNPARSEABLE: no do(var=value) found",
                                   safe_action=current_action)

        (var, value), = parsed.items()

        if var not in ACTIONABLE_VARS:
            return VerifierResult("REJECT", f"NON_ACTIONABLE: cannot intervene on '{var}'",
                                   safe_action=current_action, parsed_do=parsed)

        if not (0.0 <= value <= 1.0):
            return VerifierResult("REJECT", f"OUT_OF_RANGE: {var}={value} outside [0,1]",
                                   safe_action=current_action, parsed_do=parsed)

        cur_val = current_action.get(var, 0.5)
        clipped = False
        if abs(value - cur_val) > MAX_STEP:
            value = cur_val + np.sign(value - cur_val) * MAX_STEP
            clipped = True

        claims = parse_claims(llm_output)

        # NO-OP CHECK. Measured motivation: once the controller saturates an
        # actuator (empirically 97.5% of on-policy steps sit at a limit), the
        # model keeps proposing the value the actuator is ALREADY at, so the
        # intervention does nothing -- yet it still asserts a directional
        # effect. Sign-checking alone waves these through, because the claimed
        # direction does match the graph; it is the magnitude that is zero.
        # This was exactly the gap that capped specificity at 0.513 while
        # false rejections were already 0.000.
        effective_delta = abs(float(np.clip(value, 0.02, 0.98)) - cur_val)
        if claims and effective_delta < NO_OP_EPS:
            return VerifierResult(
                "REJECT",
                f"NO_OP_CLAIM: {var} already at {cur_val:.3f}; proposed change "
                f"{effective_delta:.4f} < {NO_OP_EPS} cannot produce the claimed effect",
                safe_action=current_action, parsed_do=parsed)

        # direction the actionable variable is actually being moved -- a claim
        # like "latency will decrease" is only checkable relative to THIS,
        # since implied_sign(var, qos_var) is the sign of d(qos_var)/d(var),
        # not the sign of the claimed outcome on its own.
        move_dir = np.sign(value - cur_val) or 1
        flagged = []
        for qos_var, claimed_sign in claims:
            implied = implied_sign(self.graph, self.edge_signs, var, qos_var)
            if implied is None:
                flagged.append((qos_var, claimed_sign, "SPURIOUS_EFFECT: no directed path in graph"))
            elif implied * move_dir != claimed_sign:
                flagged.append((qos_var, claimed_sign,
                                 f"SIGN_MISMATCH: graph implies sign {int(implied * move_dir)} "
                                 f"for this {'increase' if move_dir > 0 else 'decrease'} of {var}"))

        safe_action = dict(current_action)
        safe_action[var] = float(np.clip(value, 0.02, 0.98))

        if flagged:
            reasons = "; ".join(f"{v}: {r}" for v, _, r in flagged)
            return VerifierResult("REJECT", f"CAUSAL_CHECK_FAILED: {reasons}",
                                   safe_action=current_action, parsed_do=parsed,
                                   flagged_claims=flagged)

        if clipped:
            return VerifierResult("CLIP", f"RATE_LIMIT: clipped step to {MAX_STEP}",
                                   safe_action=safe_action, parsed_do=parsed)

        return VerifierResult("ACCEPT", "ok", safe_action=safe_action, parsed_do=parsed)
