# Author: abdkhan
"""Controlled experiments with exact ground truth, no fallback, and raw seed records.

Run with the repository's cti Python, e.g.:
  OPENBLAS_NUM_THREADS=1 python experiments/kic_controlled.py --suite all

No physical measurements are produced by this script. All licenses are scored
in shadow mode BEFORE the observation from the tick they would control.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certify.kic import CertificateBank, KICGate, KeyedRademacher

DELTA, ALPHA, STEP, TOL = .05, .05, .05, 1e-12


@dataclasses.dataclass
class CubicPlant:
    """Analytically bounded coupled plant; cost is affine in its one sensor.

    z=u-.5, m(u)=b'z + h/2 ||z||² + rho/(2p)(1'z)² + a/6 sum(z³).
    |z_j|<=.48; h=1.2, a=1.2. All action costs and dither expectations
    are analytic, so neither Monte Carlo labels nor fitted envelopes enter.
    """
    b: np.ndarray
    rho: float = .8
    h: float = 1.2
    a: float = 1.2

    @property
    def p(self):
        return len(self.b)

    @property
    def G(self):
        return float(np.max(np.abs(self.b)) + (self.h + self.rho)*.48 + self.a*.48**2/2)

    @property
    def L(self):
        # Spectral-norm bound on the Hessian; also bounds every row norm.
        return self.h + self.rho + self.a*.48

    def cost(self, u, delta=0.):
        z = np.asarray(u)-.5
        nominal = self.b @ z + self.h/2*(z @ z) + self.rho/(2*self.p)*z.sum()**2 + self.a/6*np.sum(z**3)
        return float(nominal + delta**2/2*(self.h*self.p + self.rho + self.a*z.sum()))

    def grad(self, u, delta=0.):
        z = np.asarray(u)-.5
        return self.b + self.h*z + self.rho*z.mean() + self.a/2*(z*z+delta**2)


def bank_for(p, G, L, window=None, extra=0., bounded_cv=True):
    names = [f"a{j}" for j in range(p)]
    bank = CertificateBank(names, ["cost"], DELTA, ALPHA, grid=1,
                           bounds={"cost": 2*DELTA*G+.20}, window=window,
                           robust=True, G=G, L=L, V=0., family_wise=True,
                           decision_targets=["cost"], w_extra=extra,
                           bound_control_variate=bounded_cv)
    gate = KICGate(bank, mode="hold", fallback=False, L_hat=L, cert_cap=STEP)
    return bank, gate


def full_license(bank, gate, centre, j, direction, extra=0.):
    cert = bank.query(f"a{j}", "cost", centre)
    if extra:
        cert = dataclasses.replace(cert, bias=cert.bias+extra)
    return gate.certified_step(cert, direction, -1, STEP) >= STEP-1e-14


def metric():
    return dict(beneficial=0, harmful=0, neutral=0, authorized=0,
                authorized_beneficial=0, authorized_harmful=0,
                ever_harmful=False, first_beneficial=None, worst_harm=0.)


def record(m, dm, accept, tick):
    m["beneficial"] += int(dm < -TOL)
    m["harmful"] += int(dm > TOL)
    m["neutral"] += int(abs(dm) <= TOL)
    if accept:
        m["authorized"] += 1
        m["authorized_beneficial"] += int(dm < -TOL)
        m["authorized_harmful"] += int(dm > TOL)
        m["ever_harmful"] |= bool(dm > TOL)
        m["worst_harm"] = max(m["worst_harm"], dm)
        if dm < -TOL and m["first_beneficial"] is None:
            m["first_beneficial"] = tick


def deep_bytes(obj, seen=None):
    """Retained CPython storage, including arrays; not MCU RAM or peak RSS."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_bytes(k, seen)+deep_bytes(v, seen) for k, v in obj.items())
    elif isinstance(obj, (tuple, list, set)):
        size += sum(deep_bytes(x, seen) for x in obj)
    elif hasattr(obj, "__dict__"):
        size += deep_bytes(vars(obj), seen)
    return size


def scale_job(task):
    p, rho, bounded_cv, seed, ticks = task
    rng = np.random.default_rng(10000+seed)
    plant = CubicPlant(rng.choice([-1., 1.], p)*rng.uniform(.7,.9,p), rho)
    c = rng.uniform(.35,.65,p)
    centre = {f"a{j}": float(c[j]) for j in range(p)}
    bank, gate = bank_for(p, plant.G, plant.L, bounded_cv=bounded_cv)
    exc = KeyedRademacher(f"review-scale-{seed}".encode())
    rec = {"dithered": metric(), "nominal": metric()}
    first_all = None
    # Same choices/noise across coupling strengths; forgery committed before sign.
    for tick in range(1,ticks+1):
        j, direction = int(rng.integers(p)), int(rng.choice([-1,1]))
        cp = c.copy(); cp[j] += direction*STEP
        for target, delta in [("dithered", DELTA), ("nominal", 0.)]:
            extra = plant.L*DELTA*math.sqrt(p) if target == "nominal" else 0.
            record(rec[target], plant.cost(cp,delta)-plant.cost(c,delta),
                   full_license(bank,gate,centre,j,direction,extra), tick)
        if tick % 25 == 0 and first_all is None:
            grad = plant.grad(c,DELTA)
            if all(full_license(bank,gate,centre,j,-int(np.sign(grad[j]))) for j in range(p)):
                first_all = tick
        forgery = .025*math.sin(tick/31+seed)
        signs = {f"a{j}": exc.sign(tick,j) for j in range(p)}
        u = c+DELTA*np.array(list(signs.values()))
        y = plant.cost(u)+rng.normal(0,.02)+forgery
        # Constant zero is deliberately an uninformative, predictable CV.
        # A running mean is available to all deployments and removes the level.
        bank.update(centre,signs,{"cost": y})
    return dict(p=p,rho=rho,bounded_cv=bounded_cv,seed=seed,ticks=ticks,G=plant.G,L=plant.L,
                rec=rec,first_all=first_all,certificates=bank.k_created,
                retained_bytes=deep_bytes(bank),
                dither_cost=plant.cost(c,DELTA)-plant.cost(c),
                nominal_widening=plant.L*DELTA*math.sqrt(p))


def feature(u):
    x,y=np.asarray(u)-.5
    return np.array([1,x,y,x*x,x*y,y*y,x*x*x,y*y*y])


def comparator_job(task):
    scenario, seed, ticks = task
    rng=np.random.default_rng(20000+seed)
    plant=CubicPlant(np.array([.8,-.8]))
    c=rng.uniform(.35,.65,2)
    centre={f"a{j}": float(c[j]) for j in range(2)}
    bank,gate=bank_for(2,plant.G,plant.L)
    exc=KeyedRademacher(f"review-comparator-{seed}".encode())
    poison = scenario in ("logs", "logs+online")
    spoof = scenario in ("online", "logs+online")
    # Full cubic basis is correctly specified. The comparator additionally
    # gets 1000 broad, randomized historical settings; both get live data.
    train=rng.uniform(.1,.9,(1000,2))
    X=np.stack([feature(u) for u in train])
    y=np.array([plant.cost(u) for u in train])+rng.normal(0,.02,1000)
    if poison:
        y -= 1.8*(train[:,0]-.5)
    xx,xy,yy,n=X.T@X,X.T@y,float(y@y),len(y)
    rec={k:metric() for k in ("KIC", "OLS-pointwise", "OLS-horizon", "Oracle")}
    coef=cov=None
    for tick in range(1,ticks+1):
        if coef is None or (tick-1)%25 == 0:
            inv=np.linalg.inv(xx)
            coef=inv@xy
            var=max(0.,(yy-2*coef@xy+coef@xx@coef)/(n-len(coef)))
            cov=inv*var
            crit=student_t.ppf([1-ALPHA,1-ALPHA/ticks],n-len(coef))
        j,direction=int(rng.integers(2)),int(rng.choice([-1,1]))
        cp=c.copy(); cp[j]+=direction*STEP
        # Exact expectation of the polynomial feature over all four dithers.
        dphi=np.mean([feature(cp+DELTA*np.array([s1,s2]))-feature(c+DELTA*np.array([s1,s2]))
                      for s1 in (-1,1) for s2 in (-1,1)],axis=0)
        dm=plant.cost(cp,DELTA)-plant.cost(c,DELTA)
        se=math.sqrt(max(float(dphi@cov@dphi),0.))
        decisions={"KIC":full_license(bank,gate,centre,j,direction),
                   "OLS-pointwise":float(dphi@coef)+crit[0]*se<0,
                   "OLS-horizon":float(dphi@coef)+crit[1]*se<0,
                   "Oracle":dm<0}
        for name,accept in decisions.items():
            record(rec[name],dm,accept,tick)
        signs={f"a{j}":exc.sign(tick,j) for j in range(2)}
        u=c+DELTA*np.array(list(signs.values()))
        corruption=-1.8*(c[0]-.5) if spoof else 0.
        observed=plant.cost(u)+rng.normal(0,.02)+corruption
        bank.update(centre,signs,{"cost":observed})
        f=feature(u); xx+=np.outer(f,f); xy+=f*observed; yy+=observed**2; n+=1
    return dict(scenario=scenario,seed=seed,ticks=ticks,rec=rec)


def break_job(task):
    mode,seed,ticks=task
    switch=ticks//2
    rng=np.random.default_rng(30000+seed)
    exc=KeyedRademacher(f"review-break-{seed}".encode())
    bank,gate=bank_for(1,1.2,.1,window=400,extra=2.4 if mode=="jump-budget" else 0.)
    centre={"a0":.5}
    alarm=None; first_bad=None; bad=0; pre_auth=0; last_bad=None
    interlock_delay={"interlock-0":0,"interlock-1":1,"interlock-5":5,"interlock-25":25}.get(mode)
    for tick in range(1,ticks+1):
        # At this point only samples through tick-1 are available.
        # The plant flips BEFORE tick=switch is actuated.
        slope=1.2 if tick<switch else -1.2
        if interlock_delay is not None and tick==switch+interlock_delay:
            bank.reset(); alarm=tick
        if mode=="monitor":
            iv=bank.current_slope_intervals("a0","cost",centre)
            if len(iv)>1 and max(x[0] for x in iv)>min(x[1] for x in iv):
                bank.reset()
                if alarm is None: alarm=tick
        accept=full_license(bank,gate,centre,0,-1)
        if accept and tick<switch: pre_auth+=1
        if accept and slope<0:
            bad+=1; last_bad=tick
            if first_bad is None: first_bad=tick
        s=exc.sign(tick,0)
        bank.update(centre,{"a0":s},{"cost":slope*DELTA*s+rng.normal(0,.01)},{"cost":0.})
    return dict(mode=mode,seed=seed,ticks=ticks,switch=switch,alarm=alarm,
                first_harmful=first_bad,last_harmful=last_bad,harmful=bad,
                prebreak_authorized=pre_auth)


def sensor_job(task):
    bits,rho,network,seed,ticks=task
    rng=np.random.default_rng(40000+seed)
    exc=KeyedRademacher(f"review-sensor-{seed}".encode())
    G,L=1.2,.1
    quantum=4./(2**bits)
    # Known first-order readout filter, then ADC in [-2,2]. Sending a pair
    # (q_t,q_{t-1}) permits deconvolution despite packet loss, without using
    # a plant model. Error is bounded by E, even when it depends on s_t.
    E=quantum/2*(1+rho)/(1-rho)
    bank,gate=bank_for(1,G+E/(2*DELTA),L,extra=E/DELTA)
    centre={"a0":.5}
    queue={}; received={}; filtered=0.; qprev=0.; rec=metric(); drops=expired=delivered=0
    last_processed=0
    max_error=0.
    for tick in range(1,ticks+1):
        direction=1 if rng.random()<.5 else -1
        record(rec,1.2*STEP*direction,full_license(bank,gate,centre,0,direction),tick)
        # C1 offset fixed using time only. Noise is bounded in this study.
        offset=.02*math.sin(tick/29+seed)
        s=exc.sign(tick,0)
        true=1.2*DELTA*s+rng.uniform(-.01,.01)+offset
        filtered=rho*filtered+(1-rho)*true
        assert abs(filtered)<2-quantum
        q=round(filtered/quantum)*quantum
        decoded=(q-rho*qprev)/(1-rho)
        qprev=q
        max_error=max(max_error,abs(decoded-true))
        assert abs(decoded-true)<=E+1e-12
        # Delivery choices use an independent RNG and are key independent.
        lag=int(rng.integers(0,6)) if network else 0
        lost=bool(rng.random()<.1) if network else False
        if lost: drops+=1
        elif lag>3: expired+=1
        else: queue.setdefault(tick+lag,[]).append((tick,s,decoded))
        # Authentication preserves sample IDs. Stored secret signs are matched
        # to sample IDs; no arrival-time sign or fabricated missing value.
        for sample_tick,matched_sign,value in queue.pop(tick,[]):
            received[sample_tick]=(matched_sign,value)
        # Process in ORIGINAL SAMPLE ORDER after the deadline. Consuming a
        # later filtered sample first could reveal information about an older
        # sign and invalidate the sequential independence argument.
        ready=tick-3 if network else tick
        if ready in received:
            matched_sign,value=received.pop(ready)
            assert ready>last_processed
            last_processed=ready
            bank.update(centre,{"a0":matched_sign},{"cost":value},{"cost":0.})
            delivered+=1
    pending=sum(map(len,queue.values()))+len(received)
    return dict(bits=bits,rho=rho,network=network,seed=seed,ticks=ticks,
                error_bound=E,max_error=max_error,widening=E/DELTA,
                dropped=drops,expired=expired,pending=pending,delivered=delivered,rec=rec)


def benchmark():
    """Serial host benchmark, separate from multiprocessing experiment load."""
    rows=[]
    for p in (2,4,8,16,32):
        bank,gate=bank_for(p,2.,2.6)
        centre={f"a{j}":.5 for j in range(p)}
        exc=KeyedRademacher(b"review-benchmark")
        elapsed=[]
        for t in range(1,701):
            start=time.perf_counter_ns()
            signs={f"a{j}":exc.sign(t,j) for j in range(p)}
            bank.update(centre,signs,{"cost":.02*math.sin(t)},{"cost":0.})
            full_license(bank,gate,centre,t%p,1)
            us=(time.perf_counter_ns()-start)/1000
            if t>200: elapsed.append(us)
        rows.append(dict(p=p,median_us=float(np.median(elapsed)),p95_us=float(np.quantile(elapsed,.95)),
                         retained_bytes=deep_bytes(bank),certificates=bank.k_created))
    return rows


def provenance():
    paths=[Path(__file__),ROOT/"src/certify/kic.py"]
    return dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
                hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--suite",choices=["all","scaling","comparator","break","sensor","benchmark"],default="all")
    ap.add_argument("--seeds",type=int,default=30)
    ap.add_argument("--ticks",type=int,default=1600)
    ap.add_argument("--workers",type=int,default=6)
    ap.add_argument("--out-dir",type=Path,default=ROOT/"experiments/results/controlled")
    args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    snapshot_dir=args.out_dir/'source'; snapshot_dir.mkdir(exist_ok=True)
    for path in (Path(__file__),ROOT/'src/certify/kic.py'):
        data=path.read_bytes(); digest=hashlib.sha256(data).hexdigest()
        (snapshot_dir/f'{path.stem}-{digest[:16]}.py').write_bytes(data)
    suites={
        "scaling":(scale_job,[(p,r,b,s,args.ticks) for p in (2,4,8,16,32) for r in (0.,.8)
                              for b in (False,True) for s in range(args.seeds)]),
        "comparator":(comparator_job,[(a,s,args.ticks) for a in ("clean","logs","online","logs+online") for s in range(args.seeds)]),
        "break":(break_job,[(m,s,args.ticks) for m in ("unmonitored","monitor","interlock-0","interlock-1","interlock-5","interlock-25","jump-budget") for s in range(args.seeds)]),
        "sensor":(sensor_job,[(b,r,n,s,args.ticks) for b in (16,10,6) for r in (0.,.5,.9) for n in (False,True) for s in range(args.seeds)])}
    for name,(func,jobs) in suites.items():
        if args.suite not in ("all",name): continue
        start=time.time(); rows=[]
        with ProcessPoolExecutor(args.workers) as pool:
            for row in pool.map(func,jobs):
                rows.append(row)
                if len(rows)%30==0: print(name,len(rows),"/",len(jobs),flush=True)
        payload=dict(config={**vars(args),"out_dir":str(args.out_dir)},provenance=provenance(),
                     elapsed_seconds=time.time()-start,rows=rows)
        (args.out_dir/f"{name}.json").write_text(json.dumps(payload,indent=2,allow_nan=False))
        print(name,"saved",len(rows),"runs",flush=True)
    if args.suite in ("all","benchmark"):
        (args.out_dir/"benchmark.json").write_text(json.dumps(dict(provenance=provenance(),rows=benchmark()),indent=2))


if __name__=="__main__":
    main()
