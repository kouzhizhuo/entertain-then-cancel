#!/usr/bin/env python3
"""Read the no-utterance prior-probe runs (results/a1ctx_prior/<model>/samples.jsonl), exclude scenarios the models answer 'No' (= 'not all possible')
without the utterance, write configs/a1ctx_main.json (survivors only) and results/a1ctx_prior_summary.json (counts, per-model No-rates, excluded list).
Rule (fixed before looking): a scenario is EXCLUDED if its pooled No-rate over models x {off, think_b512, think_full} x seeds is >= 0.5."""
import json, sys
from collections import defaultdict
from pathlib import Path
E = Path(__file__).resolve().parents[1]
root = Path(sys.argv[1]) if len(sys.argv) > 1 else E / "results" / "a1ctx_prior"
no = defaultdict(list); permodel = defaultdict(lambda: defaultdict(list))
for d in sorted(root.iterdir()):
    f = d / "samples.jsonl"
    if not f.exists(): continue
    for l in open(f):
        r = json.loads(l)
        if not r["valid"]: continue
        sc = int(r["item_id"].split(".")[1]); is_no = 1 if r["choice"] == 1 else 0
        no[sc].append(is_no); permodel[d.name][sc].append(is_no)
excl = sorted(sc for sc, v in no.items() if sum(v) / len(v) >= 0.5)
surv = [sc for sc in range(50) if sc not in excl]
summ = {"rule": "excluded if pooled No-rate over models x {off,b512,full} x seeds >= 0.5", "n_scenarios": 50, "n_excluded": len(excl), "excluded": excl,
        "survivors": surv, "no_rate_pooled": {sc: sum(v) / len(v) for sc, v in sorted(no.items())},
        "no_rate_by_model": {m: {sc: sum(v) / len(v) for sc, v in sorted(d.items())} for m, d in permodel.items()}}
(E / "results").mkdir(exist_ok=True)
json.dump(summ, open(E / "results" / "a1ctx_prior_summary.json", "w"), indent=1)
conds = ["off", "off_soft", "think_b0", "think_b64", "think_b128", "think_b256", "think_b512", "think_b1024", "think_b2048", "think_b256_plain", "think_b1024_plain", "think_full"]
w = {"meta.scenario": surv}
sets = []
for v, b in (("default", "a1ctx_fel"), ("ternary", "a1ctx_fel3"), ("graded", "a1ctx_fel7")):
    sets.append({"file": "a1ctx_felicity", "bucket": b, "variant": v, "where": w})
for v, b in (("default", "a1ctx_inf"), ("ternary", "a1ctx_inf3"), ("graded", "a1ctx_inf7")):
    sets.append({"file": "a1ctx_inference", "bucket": b, "variant": v, "where": w})
cfg = {"name": "a1ctx_main", "description": f"A1-ctx main on prior-probe survivors ({len(surv)}/50 scenarios; excluded {excl}).", "item_seed": 1234, "seeds": [0, 1, 2], "cap": 8192,
       "temperature": 0.6, "top_p": 0.95, "top_k": 20, "variant": "default", "conditions": conds, "sets": sets}
json.dump(cfg, open(E / "configs" / "a1ctx_main.json", "w"), indent=2)
print(f"survivors {len(surv)}/50, excluded {excl}")
