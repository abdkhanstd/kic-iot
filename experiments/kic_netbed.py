# Author: abdkhan
"""Networked emulation: does the trust boundary survive a real network path?

Three OS processes exchange real UDP datagrams on the loopback interface:

  gate   the KIC gate (the main configuration; proposer ignored, so every
         move is its own licensed fallback and any forged certificate turns
         into a harmful move), sending actuation commands and receiving
         telemetry once per tick;
  node   the IoT node: applies the actuation to the SCM plant, measures, and
         sends telemetry; an optional physical injector adds a key-blind
         additive signal to the sensor reading (C1);
  mitm   an on-path attacker between the two, with its own clock, which can
         read, rewrite, drop and replay datagrams.

The link runs either in plaintext or with AES-GCM-128 (per-direction keys,
64-bit counter nonce, the tick bound as associated data; the receiver drops
datagrams whose tag fails or whose counter does not increase). The nominal
command c_t is published on a plaintext status channel, as a command bus
would be; the applied u_t = c_t + delta s_t travels only in the actuation
datagram. Attacks:

  none           forward everything
  inject         C1: physical injection at the sensor, e_t = -0.6 (c_t - 0.5),
                 committed from the public nominal before the reading exists
  forge_aware    C2: read u_t from the actuation datagram and add
                 -0.6 (u_t - 0.5) to the telemetry
  reflect        C3: read the genuine telemetry and rewrite each QoS channel
                 as 2 * running_mean - value
  replay         C4: resend the previous tick's telemetry datagram

Scoring (after the run, from the node's log of the true regime): harmful
licensed moves on the operating cost, the operating cost itself, ticks with
no usable telemetry, and the per-datagram crypto time and size.
"""
import argparse
import json
import os
import socket
import struct
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "experiments" / "results"
HOST = "127.0.0.1"
TICK_TIMEOUT = 0.05          # loopback round trips are well below 1 ms


def ports(cfg):
    b = 40000 + 20 * cfg["run"]
    return dict(gate=b + 1, mitm_g=b + 2, mitm_n=b + 3, node=b + 4, bus=b + 5)


class Chan:
    """Plaintext or AES-GCM framing with replay protection."""

    def __init__(self, aead, key_tx, key_rx):
        self.aead = aead
        if aead:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            self.tx, self.rx = AESGCM(key_tx), AESGCM(key_rx)
        self.ctr_tx, self.ctr_rx = 0, -1
        self.crypto_us = []

    def seal(self, t, payload: bytes) -> bytes:
        if not self.aead:
            return struct.pack("!Q", t) + payload
        self.ctr_tx += 1
        nonce = struct.pack("!IQ", 0, self.ctr_tx)
        t0 = time.perf_counter()
        ct = self.tx.encrypt(nonce, payload, struct.pack("!Q", t))
        self.crypto_us.append((time.perf_counter() - t0) * 1e6)
        return struct.pack("!QQ", t, self.ctr_tx) + ct

    def open(self, pkt: bytes):
        if not self.aead:
            return struct.unpack("!Q", pkt[:8])[0], pkt[8:]
        t, ctr = struct.unpack("!QQ", pkt[:16])
        if ctr <= self.ctr_rx:
            return None                        # replayed or reordered: drop
        try:
            t0 = time.perf_counter()
            pt = self.rx.decrypt(struct.pack("!IQ", 0, ctr), pkt[16:], struct.pack("!Q", t))
            self.crypto_us.append((time.perf_counter() - t0) * 1e6)
        except Exception:
            return None                        # forged or modified: drop
        self.ctr_rx = ctr
        return t, pt


def sock(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, port))
    return s


# ------------------------------------------------------------------ node
def node_proc(cfg, keys, out: Queue):
    pt = ports(cfg)
    P_GATE, P_MITM_G, P_MITM_N, P_NODE, P_BUS = pt["gate"], pt["mitm_g"], pt["mitm_n"], pt["node"], pt["bus"]
    from certify.plants import VariantSCM
    from certify.oracle import operator_cost
    scm = VariantSCM(seed=cfg["seed"], regime_shift_prob=0.002)
    s = sock(P_NODE)
    bus = sock(P_BUS)
    s.settimeout(2.0); bus.settimeout(0.001)
    ch = Chan(cfg["aead"], keys["n2g"], keys["g2n"])
    nominal = None
    log = []
    while True:
        try:
            pkt, _ = s.recvfrom(4096)
        except socket.timeout:
            break
        if pkt == b"STOP":
            break
        # latest public nominal from the command bus (available before the tick)
        try:
            while True:
                b, _ = bus.recvfrom(64)
                nominal = struct.unpack("!Qdd", b)
        except (socket.timeout, BlockingIOError):
            pass
        m = ch.open(pkt)
        if m is None:
            continue
        t, pl = m
        u = struct.unpack("!dd", pl)
        v = scm.step({"duty_cycle": u[0], "tx_power": u[1]})
        reg = scm.regime
        log.append((t, reg.interference_mean, reg.link_quality_mean, reg.arrival_rate_mean))
        y = [v["interference"], v["link_quality"], v["arrival_rate"], v["latency"], v["energy"],
             v["packet_loss"], v["throughput"]]
        if cfg["attack"] == "inject" and nominal is not None and nominal[0] == t:
            # C1: a physical signal added at the sensor, fixed from the public
            # nominal before this tick's reading exists
            for k in (3, 4, 5):
                y[k] = float(np.clip(y[k] - 0.6 * (nominal[1] - 0.5), 0, 1))
        s.sendto(ch.seal(t, struct.pack("!7d", *y)), (HOST, P_MITM_N))
    out.put(("node", log, ch.crypto_us))


# ------------------------------------------------------------------ mitm
def mitm_proc(cfg, out: Queue):
    pt = ports(cfg)
    P_GATE, P_MITM_G, P_MITM_N, P_NODE = pt["gate"], pt["mitm_g"], pt["mitm_n"], pt["node"]
    a = sock(P_MITM_G)       # facing the gate
    b = sock(P_MITM_N)       # facing the node
    a.settimeout(0.001); b.settimeout(0.001)
    last_u, prev_tel, mean = None, None, None
    n = 0
    stop = time.time() + 3600
    while time.time() < stop:
        for sk, dst in ((a, P_NODE), (b, P_GATE)):
            try:
                pkt, _ = sk.recvfrom(4096)
            except (socket.timeout, BlockingIOError):
                continue
            if pkt == b"STOP":
                a.sendto(b"STOP", (HOST, P_NODE))
                out.put(("mitm", n))
                return
            atk, aead = cfg["attack"], cfg["aead"]
            if dst == P_NODE:                              # actuation, gate -> node
                if not aead and len(pkt) == 24:
                    last_u = struct.unpack("!dd", pkt[8:])
                sk.sendto(pkt, (HOST, dst))
                continue
            # telemetry, node -> gate
            cur = pkt
            if atk == "replay" and prev_tel is not None:
                pkt = prev_tel                             # C4: resend last tick's datagram
            elif atk in ("forge_aware", "reflect"):
                if aead:
                    pkt = pkt[:-1] + bytes([pkt[-1] ^ 1])  # any rewrite breaks the tag
                else:
                    t = struct.unpack("!Q", pkt[:8])[0]
                    y = list(struct.unpack("!7d", pkt[8:]))
                    if atk == "forge_aware" and last_u is not None:
                        for k in (3, 4, 5):
                            y[k] = float(np.clip(y[k] - 0.6 * (last_u[0] - 0.5), 0, 1))
                    else:
                        mean = y[3:6] if mean is None else [0.99 * m + 0.01 * v for m, v in zip(mean, y[3:6])]
                        for i, k in enumerate((3, 4, 5)):
                            y[k] = float(np.clip(2 * mean[i] - y[k], 0, 1))
                    pkt = struct.pack("!Q", t) + struct.pack("!7d", *y)
                n += 1
            prev_tel = cur
            sk.sendto(pkt, (HOST, dst))
    out.put(("mitm", n))


# ------------------------------------------------------------------ gate
def gate_proc(cfg, keys, out: Queue):
    pt = ports(cfg)
    P_GATE, P_MITM_G, P_BUS = pt["gate"], pt["mitm_g"], pt["bus"]
    from certify import plantspec
    from certify.kic import (KeyedRademacher, CertificateBank, KICGate, OnlineCV, ExoChangeDetector,
                             dither, EBConfSeq)
    from certify.oracle import operator_cost
    P = plantspec.get("scm")
    EBConfSeq.CAP = 0.75
    delta = 0.05
    bank = CertificateBank(P.actuators, ["cost"], delta=delta, alpha=0.05, bounds={"cost": 0.35},
                           anchor_radius=0.1, family_wise=True, decision_targets=["cost"],
                           robust=True, G=1.6, V=2e-4, L=3.5, window=400)
    gate = KICGate(bank, max_step=0.35, mode="hold", L_hat=3.5)
    cv, det = OnlineCV(["cost"], P.exo, P.actuators), ExoChangeDetector(P.exo)
    exc = KeyedRademacher(f"key-{cfg['seed']}".encode())
    s = sock(P_GATE); s.settimeout(TICK_TIMEOUT)
    bus = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ch = Chan(cfg["aead"], keys["g2n"], keys["n2g"])
    r0 = np.random.default_rng(cfg["seed"] + 99)
    nominal = {"duty_cycle": float(r0.uniform(0.1, 0.9)), "tx_power": float(r0.uniform(0.1, 0.9))}
    traj, missing, lat_ms, alarms = [], 0, [], 0
    for t in range(1, cfg["ticks"] + 1):
        for a_ in P.actuators:
            iv = bank.current_slope_intervals(a_, "cost", nominal)
            if len(iv) > 1 and max(x[0] for x in iv) > min(x[1] for x in iv):
                bank.reset(); alarms += 1
                break
        dec = gate.decide(None, nominal, "cost", sense=-1)
        nxt = {k: float(np.clip(v, 0.02 + delta, 0.98 - delta)) for k, v in dec.action.items()}
        traj.append((t, dict(nominal), dict(nxt)))
        nominal = nxt
        centre, signs, applied = dither(nominal, t, exc, delta, P.actuators)
        bus.sendto(struct.pack("!Qdd", t, nominal["duty_cycle"], nominal["tx_power"]), (HOST, P_BUS))
        t0 = time.perf_counter()
        s.sendto(ch.seal(t, struct.pack("!dd", applied["duty_cycle"], applied["tx_power"])), (HOST, P_MITM_G))
        got = None
        while True:
            try:
                pkt, _ = s.recvfrom(4096)
            except socket.timeout:
                break
            m = ch.open(pkt)
            if m is not None and m[0] == t:
                got = m
                break
            if m is not None and m[0] != t and not cfg["aead"]:
                got = (t, m[1])            # plaintext cannot tell a replay from a late datagram
                break
        if got is None:
            missing += 1
            continue
        lat_ms.append((time.perf_counter() - t0) * 1e3)
        y = struct.unpack("!7d", got[1])
        obs = dict(interference=y[0], link_quality=y[1], arrival_rate=y[2], latency=y[3], energy=y[4],
                   packet_loss=y[5], throughput=y[6], duty_cycle=applied["duty_cycle"],
                   tx_power=applied["tx_power"])
        obs["cost"] = float(operator_cost(obs))
        yh = cv.predict(obs, centre)
        cv.update(obs, centre, obs)
        if yh is not None:
            bank.update(centre, signs, obs, yh)
        if det.update(obs):
            bank.reset(); cv.reset()
    s.sendto(b"STOP", (HOST, P_MITM_G))
    out.put(("gate", traj, missing, lat_ms, ch.crypto_us, alarms))


def one(cfg):
    keys = {"g2n": os.urandom(16), "n2g": os.urandom(16)}
    q = Queue()
    ps = [Process(target=node_proc, args=(cfg, keys, q)), Process(target=mitm_proc, args=(cfg, q))]
    for p in ps:
        p.start()
    time.sleep(0.3)
    g = Process(target=gate_proc, args=(cfg, keys, q))
    g.start()
    res = {}
    for _ in range(3):
        m = q.get(timeout=3600)
        res[m[0]] = m[1:]
    for p in ps + [g]:
        p.join(timeout=10)
    # score against the node's log of the true regime
    from certify import plantspec
    from simulator.scm_iot import RegimeParams
    P = plantspec.get("scm")
    reg_at = {t: RegimeParams(interference_mean=i, link_quality_mean=l, arrival_rate_mean=a)
              for t, i, l, a in res["node"][0]}
    traj, missing, lat, cus, alarms = res["gate"]
    harm = moves = 0
    ecost = []
    maxh = 0.0
    for t, c0, c1 in traj:
        reg = reg_at.get(t) or reg_at.get(t - 1)
        if reg is None:
            continue
        m1 = P.op_cost(reg, c1, 0.05, n=2000)
        ecost.append(m1)
        if c1 != c0:
            dh = m1 - P.op_cost(reg, c0, 0.05, n=2000)
            moves += 1; harm += dh > 1e-3; maxh = max(maxh, dh)
    return dict(cfg, moves=moves, harmful=harm, max_harm=maxh, ecost=float(np.mean(ecost)),
                missing=missing, alarms=alarms, rtt_ms_median=float(np.median(lat)) if lat else None,
                crypto_us_median=float(np.median(cus + res["node"][1])) if (cus or res["node"][1]) else None)


def _one(cfg):
    return one(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=800)
    ap.add_argument("--par", type=int, default=10)
    a = ap.parse_args()
    cfgs = [dict(aead=aead, attack=atk, seed=sd, ticks=a.ticks)
            for aead in (False, True) for atk in ("none", "inject", "forge_aware", "reflect", "replay")
            for sd in range(a.seeds)]
    for i, c in enumerate(cfgs):
        c["run"] = i
    from multiprocessing.pool import ThreadPool
    with ThreadPool(a.par) as tp:
        rows = tp.map(_one, cfgs)
    for r in rows:
        print(json.dumps(r), flush=True)
    agg = {}
    for aead in (False, True):
        for atk in ("none", "inject", "forge_aware", "reflect", "replay"):
            R = [r for r in rows if r["aead"] == aead and r["attack"] == atk]
            agg[f"{'aead' if aead else 'plain'}|{atk}"] = dict(
                ecost=float(np.mean([r["ecost"] for r in R])), harmful=sum(r["harmful"] for r in R),
                moves=sum(r["moves"] for r in R), max_harm=max(r["max_harm"] for r in R),
                missing=float(np.mean([r["missing"] for r in R])),
                rtt_ms=float(np.median([r["rtt_ms_median"] for r in R if r["rtt_ms_median"]])),
                crypto_us=(float(np.median([r["crypto_us_median"] for r in R if r["crypto_us_median"]]))
                           if aead else None))
    for k, v in agg.items():
        print(k, v)
    json.dump(dict(agg=agg, rows=rows), open(RES / "kic_netbed.json", "w"), indent=1)


if __name__ == "__main__":
    main()
