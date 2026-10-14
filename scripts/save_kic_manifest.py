# Author: abdkhan
"""Pin every result file and source the KIC paper depends on (SHA256).

This manifest pins what the paper's tables and text are generated from: experiments/results (every .json
and .csv the generators read), the gate and simulators (src/), the C port
(firmware/), the experiment scripts and the tests.

Text files are hashed with CRLF normalized to LF, so a checkout on Windows and
one on Linux verify against the same digests.

Usage:
    python scripts/save_kic_manifest.py            # write experiments/KIC_MANIFEST.json
    python scripts/save_kic_manifest.py --verify   # re-hash and diff
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "KIC_MANIFEST.json"
TEXT = {".py", ".json", ".csv", ".tex", ".bib", ".c", ".h", ".sh", ".md", ".txt", ".tikz", ".resc", ".repl", ".ld"}
GROUPS = {
    "results": ["experiments/results/*.json", "experiments/results/*.csv",
                "experiments/results/pil_base/*.json", "experiments/results/controlled/*.json"],
    "gate_and_plants": ["src/certify/*.py", "src/simulator/*.py", "src/attacks/telemetry_poisoning.py"],
    "c_port": ["firmware/src/*", "firmware/*.py"],
    "experiments": ["experiments/kic_*.py", "experiments/mec_world.py", "experiments/grid_world.py",
                    "experiments/benchmark_kic_bank.py", "experiments/run_*.sh"],
    "tests": ["tests/*.py"],
}


def digest(p: Path) -> str:
    b = p.read_bytes()
    if p.suffix.lower() in TEXT:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def build() -> dict:
    out = {}
    for group, pats in GROUPS.items():
        files = sorted({f for pat in pats for f in ROOT.glob(pat) if f.is_file()})
        out[group] = {f.relative_to(ROOT).as_posix(): digest(f) for f in files}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    now = build()
    if not a.verify:
        MANIFEST.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST.relative_to(ROOT)}: " + ", ".join(f"{g} {len(v)}" for g, v in now.items()))
        return
    old = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for g in sorted(set(old) | set(now)):
        o, n = old.get(g, {}), now.get(g, {})
        for f in sorted(set(o) | set(n)):
            if o.get(f) != n.get(f):
                bad += 1
                print(f"{g}: {f}: " + ("missing" if f not in n else "new" if f not in o else "changed"))
    print("manifest OK" if not bad else f"{bad} file(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
