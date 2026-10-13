#!/usr/bin/env python3
"""Merge the per-dose frontier call files (results/api/<provider_model>/<dose>/samples.jsonl) into exporter-readable runs:
results/api_runs/E1f/<model>/{samples.jsonl,results.json} (condition = dose token), then export with
    python3 analysis/export_e1.py e1fr results/api_runs results/E1F        # keys e1f.<model>.<set>.<fmt>.<dose>.<metric>.<stat>
Run:  python3 analysis/api_finalize.py && python3 analysis/export_e1.py e1fr results/api_runs results/E1F"""
import json
from pathlib import Path
X = Path(__file__).resolve().parents[1] / "results"; OUT = X / "api_runs" / "E1f"
for md in sorted((X / "api").glob("*")):
    if not md.is_dir(): continue
    rows, doses, models, dates = [], [], set(), set()
    for dd in sorted(md.glob("*")):
        f = dd / "samples.jsonl"
        if not f.exists(): continue
        rs = [json.loads(l) for l in open(f)]
        if not rs: continue
        rows += rs; doses.append(dd.name); models |= {r.get("model_returned") for r in rs}; dates |= {r["date"] for r in rs}
    if not rows: continue
    name = md.name.split("_", 1)[1]                         # openai_gpt-5.4 -> gpt-5.4
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    with open(d / "samples.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    usd = sum((r.get("usd") or 0) for r in rows)
    json.dump({"meta": {"model_name": name, "model_id": md.name.replace("_", "/", 1), "conditions": doses, "seeds": sorted({r["seed"] for r in rows}), "n_items": len({r["item_id"] for r in rows}),
                        "n_samples": len(rows), "returned_model_ids": sorted(m for m in models if m), "dates": sorted(dates), "usd": round(usd, 4),
                        "usd_estimated_rows": sum(1 for r in rows if r.get("usd_est")), "sampling": "provider default temperature (not settable for the reasoning arms); 1 sample per item; letter forced choice",
                        "note": "closed models can change; IDs and dates recorded; subjects only, never judges"}, "aggregate": {}}, open(d / "results.json", "w"), indent=1)
    print(name, doses, len(rows), "calls", f"${usd:.2f}")
