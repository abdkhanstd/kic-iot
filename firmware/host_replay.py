# Author: abdkhan
"""Native (host) replay of the C gate against the Python reference.

For each scenario and seed, record the Python closed loop of the fully
licensed KIC gate, compile the C port with a host driver, replay the trace
tick by tick and count decisions that differ by more than 1e-9. No emulator
is needed; firmware/pil.py adds the instruction-set emulation.
"""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import pil
from pathlib import Path
pil.BUILD.mkdir(exist_ok=True)
out = {}
for scen in ["clean", "backdoor_lie", "a1_telemetry", "spoof_blind", "oracle"]:
    for seed in [0, 1, 2, 3]:
        T = pil.make_trace(scen, seed, 800)
        hdr = pil.BUILD / f"hdr_host_{scen}_{seed}"; hdr.mkdir(exist_ok=True)
        pil.write_header(T, hdr / "trace.h")
        h = pil.build_host(hdr).strip()
        out[f"{scen}|{seed}"] = h
        print(scen, seed, h, flush=True)
json.dump(out, open(pil.RES / 'kic_host_replay.json', 'w'), indent=1)
