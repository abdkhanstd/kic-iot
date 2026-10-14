# Author: abdkhan
"""Noise tail on the feeder: form (ii) of Corollary cor:tail (supplement, feeder section).

Off its certified domain D_f the feeder has no derivative bound (a load far beyond 5 sigma can
drive the sweep towards collapse), but its channels are range-limited, so on every realization
  |D_t| <= delta * Gbar    and    |J(c') - J(c)| <= Rbar  for one-actuator moves up to MAX_STEP.
The certificates then cover the slope of the tail-free cost M^c with 2*Gbar*eps of the runs'
widening w_extra = 0.016, and a licensed move lowers M itself only if its certified decrease of
M^c exceeds eps*Rbar.  This script re-runs the stored level-T feeder configurations
(kic_shadow_grid_cert.json, kic_certified_grid_D.json) with every executed KIC move logged:
step Delta, slack s (recomputed with 2*Gbar*eps in place of w_extra) and the certified decrease
s*Delta - L Delta^2/2, against eps*Rbar(Delta).

  python experiments/kic_tail_check.py run shadow DIR_S [--procs 12]
  python experiments/kic_tail_check.py run closed:learned DIR_L [--procs 12]
  python experiments/kic_tail_check.py run closed:spoof_blind DIR_B [--procs 12]
  python experiments/kic_tail_check.py summarize DIR_S DIR_L DIR_B   -> results/kic_tail_check.json

The shadow re-run stubs the Monte-Carlo ground truth (shadow proposals are random, so it never
feeds back into a decision). In closed loop, KIC's trajectory is identical in every seed across the
five proposer scenarios (same cost, moves and first license in the stored runs), and across the two
spoofing scenarios; the summary verifies this from the stored rows, re-runs one scenario of each
group (neither uses the oracle's Monte-Carlo planner), and checks every decision against the stored runs.
"""
import json, os, sys
from pathlib import Path

# one BLAS thread per worker and below-normal priority (inherited by spawned workers on Windows)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
if os.name == "nt":
    import ctypes
    _k = ctypes.windll.kernel32
    _k.GetCurrentProcess.restype = ctypes.c_void_p
    _k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _k.SetPriorityClass(_k.GetCurrentProcess(), 0x4000)          # BELOW_NORMAL_PRIORITY_CLASS
os.environ["KIC_GRID_DOMAIN"] = "1"
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments" / "results"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
# spawned workers re-run this module with the run script's argv: settings travel in the environment
if "TAIL_OUT" not in os.environ and len(sys.argv) > 3 and sys.argv[1] == "run":
    os.environ["TAIL_MODE"], os.environ["TAIL_OUT"] = sys.argv[2], str(Path(sys.argv[3]).resolve())
MODE, OUT = os.environ.get("TAIL_MODE"), Path(os.environ.get("TAIL_OUT", "."))
from scipy.stats import norm
from certify import kic as K
from certify import grid as GR

EPS = 69 * 2 * norm.cdf(-5)                      # some load, PV or meter noise beyond 5 sigma
DELTA, W_EXTRA, MAX_STEP = 0.05, 0.016, 0.35
QMAXABS = max(abs(GR._q_of(K.LO - DELTA)), abs(GR._q_of(K.HI + DELTA)))
R_CH = GR.COST_W["loss"] * (GR.LOSS_HI - GR.LOSS_LO) + 2 * GR.COST_W["v"] * (GR.V_HI - GR.V_LO)
GBAR = 0.5 * (R_CH + GR.COST_W["q"] * 1.4 * 2 * DELTA * 2 * QMAXABS) / DELTA
RBAR = lambda d: R_CH + GR.COST_W["q"] * 1.4 * d * 2 * QMAXABS
GBAR_PRINT, RBAR_PRINT = 131.0, 13.4             # values printed in the supplement (rounded up)

_cs, _q, _dec, _fb = K.KICGate.certified_step, K.CertificateBank.query, K.KICGate.decide, K.KICGate._fallback
_feeder = lambda gate: abs(gate.bank.w_extra - W_EXTRA) < 1e-12


def q(self, actuator, target, centre):
    c = _q(self, actuator, target, centre)
    c._act = actuator
    return c


def cs(self, cert, d, sense, cap):
    r = _cs(self, cert, d, sense, cap)
    if r > 0 and cert.robust and _feeder(self):
        m = -cert.hi if d > 0 else cert.lo
        if sense > 0:
            m = cert.lo if d > 0 else -cert.hi
        slack = (m - cert.bias - self.L_hat * (cert.eta + 0.5 * self.bank.delta) - self.bank.V * (cert.age + 1))
        self._rec.setdefault(getattr(cert, "_act", None), []).append((d, slack))
    return r


def log(self, current, dec):
    if dec.verdict not in ("ACCEPT", "FALLBACK") or not _feeder(self):
        return dec
    ch = [a for a in self.bank.actuators if abs(dec.action[a] - current[a]) > 1e-12]
    if not ch:
        return dec
    a = ch[0]; D = abs(dec.action[a] - current[a]); d = 1 if dec.action[a] > current[a] else -1
    recs = [s for (dd, s) in self._rec.get(a, []) if dd == d]
    rec = dict(v=dec.verdict, D=D, s_code=recs[-1] if recs else None, L=self.L_hat)
    with open(OUT / f"rec_{os.getpid()}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return dec


def decide(self, proposal, current, objective, sense=-1):
    self._rec = {}
    return log(self, current, _dec(self, proposal, current, objective, sense))


def fallback(self, current, objective, sense, reason, cert=None):
    if not hasattr(self, "_rec"):
        self._rec = {}
    return _fb(self, current, objective, sense, reason, cert)


if MODE == "shadow" or (MODE or "").startswith("closed:"):
    K.CertificateBank.query, K.KICGate.certified_step = q, cs
    K.KICGate.decide, K.KICGate._fallback = decide, fallback
if MODE == "shadow":
    from certify import plantspec as _PS
    _PS.PlantSpec.op_cost = lambda self, *a, **k: 0.0


GROUPS = {"learned": ["oracle", "oracle_backdoor", "learned", "backdoor_lie", "a1_telemetry"],
          "spoof_blind": ["spoof_blind", "a1_spoof"]}


def summarize(dshadow, dlearned, dspoof):
    load = lambda d: [json.loads(l) for f in sorted(Path(d).glob("rec_*.jsonl")) for l in open(f)]
    out = dict(eps=EPS, Gbar=GBAR, Rbar_max=RBAR(MAX_STEP), Gbar_print=GBAR_PRINT, Rbar_print=RBAR_PRINT,
               two_Gbar_eps=2 * GBAR_PRINT * EPS, eps_Rbar=EPS * RBAR_PRINT, w_extra=W_EXTRA, summary={})
    ok = True
    st, sr = json.load(open(Path(dshadow) / "shadow.json"))["agg"], json.load(open(RES / "kic_shadow_grid_cert.json"))["agg"]
    keys = [k for k in sr if k.split("|")[2] in ("KIC", "KIC-pc")]
    ok &= all(st[k]["auth"] == sr[k]["auth"] for k in keys)
    stored = json.load(open(RES / "kic_certified_grid_D.json"))
    cr = stored["agg"]
    rows = [r for r in stored["rows"] if r["gate"] == "kic_sound"]
    for rep_, grp in GROUPS.items():     # identical trajectories within each group, seed by seed
        for sd in {r["seed"] for r in rows}:
            ok &= len({(r["ecost"], r["moves"], r["first_licensed"]) for r in rows
                       if r["seed"] == sd and r["scenario"] in grp}) == 1
    runs = [("shadow (KIC, KIC-pc; 4 scenarios)", dshadow, 1, sum(sr[k]["auth"] for k in keys))]
    for rep_, d in (("learned", dlearned), ("spoof_blind", dspoof)):
        ct = json.load(open(Path(d) / f"closed_{rep_}.json"))["agg"]
        k = f"grid|{rep_}|kic_sound"
        ok &= ct[k]["moves"] == cr[k]["moves"] and abs(ct[k]["ecost"] - cr[k]["ecost"]) < 1e-12
        runs.append((f"closed loop, {rep_} (kic_sound; stands for {len(GROUPS[rep_])} scenarios)", d,
                     len(GROUPS[rep_]), cr[k]["moves"]))
    for name, d, w, n_stored in runs:
        R = load(d)
        ok &= len(R) == n_stored and all(r["s_code"] is not None for r in R)
        for r in R:   # certified decrease of M^c: slack with 2*Gbar*eps in place of w_extra (L = 7.5 on the feeder)
            s_true = r["s_code"] + W_EXTRA - 2 * GBAR_PRINT * EPS
            r["gain"] = s_true * r["D"] - 0.5 * r.get("L", 7.5) * r["D"] ** 2
        clear = [r for r in R if r["gain"] > EPS * RBAR_PRINT]
        out["summary"][name] = dict(n=len(R), n_stored=n_stored, weight=w, clear=len(clear), below=len(R) - len(clear),
                                    licenses=sum(r["v"] == "ACCEPT" for r in R), fallbacks=sum(r["v"] == "FALLBACK" for r in R),
                                    median_step=sorted(r["D"] for r in R)[len(R) // 2],
                                    min_gain=min(r["gain"] for r in R), max_gain=max(r["gain"] for r in R))
    out["decisions_match"] = bool(ok)
    json.dump(out, open(RES / "kic_tail_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "summarize":
        summarize(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        procs = sys.argv[sys.argv.index("--procs") + 1] if "--procs" in sys.argv else "12"
        OUT.mkdir(parents=True, exist_ok=True)
        if MODE == "shadow":
            import kic_shadow as M
            sys.argv = ["kic_shadow.py", "--seeds", "15", "--dwell", "800", "--n_dwell", "1", "--stationary", "--certified",
                        "--plants", "grid", "--scenarios", "none", "a1_telemetry", "spoof_blind", "a1_spoof",
                        "--procs", procs, "--out", "tailcheck_shadow.json"]
        else:
            import kic_certified as M
            M.GATES, M.SCEN["grid"] = ["kic_sound"], [MODE.split(":")[1]]
            sys.argv = ["kic_certified.py", "--seeds", "15", "--plants", "grid", "--procs", procs,
                        "--out", f"tailcheck_{MODE.replace(':', '_')}.json"]
        M.main()
        # the run scripts write into results/ (and prefix raw_); move their output next to the records
        import shutil
        tag = MODE.replace(":", "_")
        for f in RES.glob(f"*tailcheck_{tag}.json"):
            shutil.move(str(f), str(OUT / f.name.replace(f"tailcheck_{tag}", tag)))
