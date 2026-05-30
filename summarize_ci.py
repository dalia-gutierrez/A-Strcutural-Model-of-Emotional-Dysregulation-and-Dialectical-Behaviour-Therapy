"""Print the 90% welfare confidence interval from each bootstrap file."""
import json, numpy as np, os
MIDUS = 46072.0
for fn, label in [("boot_meta.jsonl",  "raw-cross-section prior (chi~0.158, B~3.92)"),
                  ("boot_table1.jsonl","Table-1 prior (chi~0.024, B~3.85)"),
                  ("boot.jsonl",       "single-EMA alpha (no meta-anchor)")]:
    if not os.path.exists(fn):
        continue
    a = np.array([json.loads(l) for l in open(fn)]) * MIDUS
    print(f"{label:48s}  n={len(a):3d}  "
          f"median=${np.median(a):,.0f}  "
          f"90% CI=[${np.percentile(a,5):,.0f}, ${np.percentile(a,95):,.0f}]")
