#!/usr/bin/env python3
"""Seed-concatenating merge (LOCAL): for (base group, extra-seeds group) pairs present in the merge tree, replace the base model dir by a real dir whose samples.jsonl is
base (seeds 0-2) + extra (seeds 3-5) and whose results.json meta lists the union of seeds/conditions, so a cell has 6 seeds. usage: concat_seeds.py <tree>"""
import json, shutil, sys
from pathlib import Path
T = Path(sys.argv[1]); PAIRS = [("E3_gptoss", "E3_gptoss_s2"), ("E1s", "E1s_s2"), ("E1b", "E1b_s2"), ("E1q", "E1q_s2")]
for base, extra in PAIRS:
    for ed in sorted((T / extra).glob("*")) if (T / extra).exists() else []:
        bd = T / base / ed.name
        if not bd.exists() or not (ed / "samples.jsonl").exists(): continue
        bsrc = bd.resolve(); out = T / "_cat" / base / ed.name; out.mkdir(parents=True, exist_ok=True)
        rows = [l for l in open(bsrc / "samples.jsonl")] + [l for l in open(ed.resolve() / "samples.jsonl")]
        seeds = sorted({json.loads(l)["seed"] for l in rows}); ids = {(json.loads(l)["item_id"], json.loads(l)["bucket"], json.loads(l)["condition"], json.loads(l)["seed"]) for l in rows}   # item ids repeat across buckets (felicity / NLI framings of the same item)
        assert len(ids) == len(rows), f"duplicate (item, condition, seed) in {base}/{ed.name}"
        open(out / "samples.jsonl", "w").writelines(rows)
        R = json.load(open(bsrc / "results.json")); R2 = json.load(open(ed.resolve() / "results.json"))
        R["meta"]["seeds"] = seeds; R["meta"]["conditions"] = sorted(set(R["meta"]["conditions"]) | set(R2["meta"]["conditions"]), key=R["meta"]["conditions"].index if False else str)
        R["meta"]["n_samples"] = len(rows); R["aggregate"] = {}; R["meta"]["concatenated_from"] = [str(bsrc), str(ed.resolve())]
        json.dump(R, open(out / "results.json", "w"), indent=1)
        (T / base / ed.name).unlink(); (T / base / ed.name).symlink_to(out)
        print(f"concat {base}/{ed.name}: seeds {seeds}, {len(rows)} rows")
