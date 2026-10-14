# Author: abdkhan
"""Invariants the paper's claims rest on. Run: python -m pytest tests/ (or python tests/test_kic.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.scm_iot import IoTClusterSCM, RegimeParams
from certify.kic import KeyedRademacher, EBConfSeq, CertificateBank, dither
from certify.oracle import expected_outcome, true_local_effect
from certify.loop import with_cost, ACTUATORS


def test_oracle_matches_simulator():
    scm = IoTClusterSCM(seed=1)
    a = dict(duty_cycle=0.4, tx_power=0.6)
    cf = scm.counterfactual(a, do={}, n_samples=4000)
    for k in ["latency", "energy", "packet_loss", "throughput"]:
        assert abs(cf[k] - expected_outcome(RegimeParams(), a, k, n=20000)) < 0.01


def _keyed_estimate(forgery, n=6000, cv_bias=0.0):
    """Mean of s*(y - yhat)/delta for duty_cycle -> latency at a fixed action."""
    scm = IoTClusterSCM(seed=0, regime_shift_prob=0.0)
    exc, delta = KeyedRademacher(b"test"), 0.05
    nom = dict(duty_cycle=0.5, tx_power=0.5)
    acc = []
    for t in range(1, n + 1):
        c, s, u = dither(nom, t, exc, delta, ACTUATORS)
        y = scm.step(u)["latency"] + forgery(t, s, u)
        yhat = 0.6 + cv_bias * np.sin(t)          # any predictor fixed before s_t
        acc.append(s["duty_cycle"] * (y - yhat) / delta)
    return float(np.mean(acc)), float(np.std(acc) / np.sqrt(n))


def test_keyed_estimator_unbiased_and_forgery_invariant():
    truth = true_local_effect(RegimeParams(), dict(duty_cycle=0.5, tx_power=0.5),
                              "duty_cycle", "latency", 0.05, n=20000)
    est, se = _keyed_estimate(lambda t, s, u: 0.0)
    assert abs(est - truth) < 4 * se + 0.02
    # key-blind forgery (depends on t only) and an adversarial CV: unchanged
    est2, se2 = _keyed_estimate(lambda t, s, u: 0.3 * np.cos(t), cv_bias=0.5)
    assert abs(est2 - truth) < 4 * se2 + 0.02
    # key-aware forgery (reads the applied actuator): biased, as Theorem 1 states
    est3, _ = _keyed_estimate(lambda t, s, u: -2.0 * (u["duty_cycle"] - 0.5))
    assert est3 < truth - 1.0


def test_eb_cs_covers_null():
    rng = np.random.default_rng(0)
    fails = 0
    for r in range(200):
        cs = EBConfSeq(0.05)
        bad = False
        for _ in range(500):
            cs.update(float(rng.random() < 0.5))
            lo, hi = cs.interval()
            bad |= not (lo <= 0.5 <= hi)
        fails += bad
    assert fails / 200 <= 0.05 + 0.03


def test_upper_mean_cs_covers():
    """UpperMeanCS with externally supplied predictable weights upper-bounds
    the weighted frequency of a Bernoulli(p) stream uniformly in time."""
    from certify.kic import UpperMeanCS
    rng = np.random.default_rng(1)
    fails = 0
    for r in range(200):
        u, bad = UpperMeanCS(0.05), False
        for t in range(1, 400):
            u.update(float(rng.random() < 0.02), lam=min(0.75, 1.0 / np.sqrt(t)))
            bad |= u.upper() < 0.02
        fails += bad
    assert fails / 200 <= 0.05 + 0.03


def test_rms_dist_matches_direct():
    """The lambda-weighted RMS distance from the running moments equals the
    direct computation (moments are kept about the first location)."""
    rng = np.random.default_rng(2)
    cs = EBConfSeq(0.05)
    locs, lams = [], []
    for t in range(200):
        loc = np.array([0.5, 0.4]) + 0.05 * rng.standard_normal(2)
        lams.append(cs.next_lam())
        cs.update(float(rng.random()), tau=t, loc=loc)
        locs.append(loc)
    x = np.array([0.52, 0.38])
    L, w = np.array(locs), np.array(lams)
    direct = np.sqrt((w * ((L - x) ** 2).sum(1)).sum() / w.sum())
    assert abs(cs.rms_dist(x) - direct) < 1e-10
    assert abs(cs.mean_age(200) - (200 - (w * np.arange(200)).sum() / w.sum())) < 1e-9


def test_current_slope_certificate_under_drift():
    """Theorem 3: with the declared variation rate, the widened interval
    contains the CURRENT slope of a linearly drifting response at every tick."""
    from certify.kic import CertificateBank, KeyedRademacher
    T, S, delta = 1200, 1.0, 0.05
    V = 2 * S / T
    fails = 0
    for r in range(40):
        rng = np.random.default_rng(100 + r)
        exc = KeyedRademacher(f"t{r}".encode())
        bank = CertificateBank(["a"], ["y"], delta, 0.05, grid=1, bounds={"y": 0.35}, window=150,
                               robust=True, G=1.5, V=V, L=0.0)
        bad = False
        for t in range(1, T + 1):
            s = exc.sign(t, 0)
            slope = S - 2 * S * t / T
            bank.update({"a": 0.5}, {"a": s}, {"y": s * delta * slope + rng.normal(0, 0.03)}, {"y": 0.0})
            c = bank.query("a", "y", {"a": 0.5})
            w = c.bias + V * (c.age + 1)
            bad |= not (c.lo - w <= S - 2 * S * (t + 1) / T <= c.hi + w)
        fails += bad
    assert fails / 40 <= 0.05 + 0.1


def test_simultaneous_dither_targets_operating_cost():
    """Counterexample m(u1, u2) = u1 * u2^2 with both actuators
    dithered at once: the keyed estimand for actuator 1 is the slope of the
    operating cost M(c) = E_s m(c + delta s) (Lemma 1), c2^2 + delta^2, and
    not the nominal slope c2^2."""
    rng = np.random.default_rng(3)
    exc, delta, n = KeyedRademacher(b"estimand"), 0.2, 40000
    c1, c2 = 0.5, 0.5
    acc = np.empty(n)
    for t in range(n):
        s1, s2 = exc.sign(t, 0), exc.sign(t, 1)
        u1, u2 = c1 + delta * s1, c2 + delta * s2
        y = u1 * u2 ** 2 + rng.normal(0, 0.01)
        # as in CertificateBank.update: a predictable baseline and the other
        # actuator's dither removed with a (here exact) past slope estimate;
        # both are independent of s1, so they change the variance only
        r = y - c1 * c2 ** 2 - delta * s2 * 2 * c1 * c2
        acc[t] = s1 * r / delta
    est, se = acc.mean(), acc.std() / np.sqrt(n)
    slope_M, slope_m = c2 ** 2 + delta ** 2, c2 ** 2
    assert abs(est - slope_M) < 4 * se
    assert abs(est - slope_m) > 8 * se


def test_operating_cost_matches_plantspec():
    """plantspec.local_effect(smoothed=True) is the Lemma-1 estimand and
    op_cost is the dither-averaged cost; the difference quotient of op_cost
    along actuator j agrees with the smoothed estimand to O(delta^2)."""
    from certify import plantspec
    P = plantspec.get("scm")
    reg = RegimeParams()
    c = dict(duty_cycle=0.45, tx_power=0.55)
    b = P.local_effect(reg, c, "duty_cycle", "cost", 0.05, n=20000)
    h = 1e-3
    up = dict(c); up["duty_cycle"] += h
    dn = dict(c); dn["duty_cycle"] -= h
    dM = (P.op_cost(reg, up, 0.05, n=20000) - P.op_cost(reg, dn, 0.05, n=20000)) / (2 * h)
    # |beta - d_j M| <= L delta / 2 (Theorem 3's bridging term); L <= 3.5 here
    assert abs(b - dM) <= 3.5 * 0.05 / 2


def test_nominal_is_dither_centre_even_for_edge_proposals():
    """Algorithm 1 invariant: every KIC nominal lies in C = [0.02+delta,
    0.98-delta]^p, so the applied action is nominal +- delta exactly and the
    certificate update, the operating cost and the harm score are indexed by
    the same point, even when the proposer asks for the range edges."""
    from certify.loop import RunConfig, run

    class EdgeProposer:
        def propose(self, nominal, obs, objective, t, ctx):
            a = ACTUATORS[t % 2]
            v = 0.0 if (t // 2) % 2 else 1.0
            return {a: v}, f"set {a} to {v}"

    delta = 0.05
    trace = []
    run(RunConfig(gate="kic_sound", n_steps=400, seed=3, delta=delta), EdgeProposer(), pil_trace=trace)
    assert len(trace) == 400
    for row in trace:
        for c, u in zip(row["nominal"], row["applied"]):
            assert 0.02 + delta - 1e-12 <= c <= 0.98 - delta + 1e-12
            assert abs(abs(u - c) - delta) < 1e-12


if __name__ == "__main__":
    for f in [test_oracle_matches_simulator, test_keyed_estimator_unbiased_and_forgery_invariant,
              test_eb_cs_covers_null, test_upper_mean_cs_covers, test_rms_dist_matches_direct,
              test_current_slope_certificate_under_drift,
              test_simultaneous_dither_targets_operating_cost, test_operating_cost_matches_plantspec,
              test_nominal_is_dither_centre_even_for_edge_proposals]:
        f(); print("ok", f.__name__)
