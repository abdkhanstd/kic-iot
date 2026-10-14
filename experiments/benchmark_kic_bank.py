# Author: abdkhan
"""Serial host measurements of certificate-bank growth; no hardware claim."""
import json
import time
from pathlib import Path

import numpy as np
from kic_controlled import bank_for, deep_bytes, full_license, KeyedRademacher, provenance


def main():
    rows=[]
    for p in (2,4,8,16,32):
        for anchors in (1,4,16):
            bank,gate=bank_for(p,2.,2.6)
            bank.anchor_radius=.02
            centres=[{f"a{j}":(.1+.05*k if j==0 else .5) for j in range(p)} for k in range(anchors)]
            exc=KeyedRademacher(b"bank-growth")
            elapsed=[]
            for tick in range(1,100*anchors+501):
                centre=centres[tick%anchors]
                start=time.perf_counter_ns()
                signs={f"a{j}":exc.sign(tick,j) for j in range(p)}
                bank.update(centre,signs,{"cost":.02*np.sin(tick)},{"cost":0.})
                full_license(bank,gate,centre,tick%p,1)
                us=(time.perf_counter_ns()-start)/1000
                if tick>100*anchors:elapsed.append(us)
            assert len(bank._anchors)==anchors
            rows.append(dict(p=p,anchors=anchors,certificates=bank.k_created,
                             median_us=float(np.median(elapsed)),p95_us=float(np.quantile(elapsed,.95)),
                             retained_bytes=deep_bytes(bank)))
    path=Path(__file__).resolve().parent/'results/controlled/bank_growth.json'
    path.write_text(json.dumps(dict(provenance=provenance(),rows=rows),indent=2))
    print(path)


if __name__=='__main__':main()
