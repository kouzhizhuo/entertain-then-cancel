#!/usr/bin/env python3
"""Aggregate per-sample JSONL into results.json: rates with mean ± std over seeds and item-bootstrap 95% CIs.

usage: python aggregate.py samples.jsonl [results.json]

Metrics per (group, condition):
  gold_rate     fraction of answers equal to the gold reading (= the pragmatic reading whenever the source defines one)
  literal_rate  fraction equal to the literal-reading option (only for items where the source defines one)
  invalid_rate  answers with no valid option letter
  tokens        mean thinking tokens / total generated tokens; forced_rate = share of traces cut by the budget
'std' is the std over seeds of the per-seed rate (ddof=1, seeds >= 2), 'ci' the 95% percentile bootstrap over items
of the seed-averaged per-item rate. Groups: phenomenon, phenomenon/subset, bucket, role.
"""
import json, sys
from collections import defaultdict
from statistics import mean, stdev

import numpy as np


def _boot_ci(per_item, n=2000, seed=0):
    x = np.asarray(per_item, float)
    if len(x) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        for g in (f"bucket:{r['bucket']}", f"phen:{r['phenomenon']}", f"sub:{r['phenomenon']}/{r['subset']}",
                  f"bucket:{r['bucket']}|role:{r['role']}"):
            groups[(g, r["condition"])].append(r)
    out = {}
    for (g, cond), rs in sorted(groups.items()):
        seeds = sorted({r["seed"] for r in rs})
        items = sorted({r["item_id"] for r in rs})
        cell = {"n_items": len(items), "n_samples": len(rs), "seeds": seeds}
        for metric, key in (("gold_rate", "is_gold"), ("literal_rate", "is_literal")):
            sel = [r for r in rs if r[key] is not None]
            if not sel:
                continue
            per_seed = [mean(r[key] for r in sel if r["seed"] == s) for s in seeds if any(r["seed"] == s for r in sel)]
            per_item = defaultdict(list)
            for r in sel:
                per_item[r["item_id"]].append(r[key])
            m = mean(r[key] for r in sel)
            cell[metric] = {"mean": m, "std": stdev(per_seed) if len(per_seed) > 1 else None,
                            "ci": _boot_ci([mean(v) for v in per_item.values()]), "n_items": len(per_item), "per_seed": per_seed}
        rt = [r["rating_exp"] for r in rs if r.get("rating_exp") is not None]
        if rt:
            cell["rating_exp_mean"] = mean(rt)
            cell["rating_argmax_mean"] = mean(r["rating"] for r in rs if r.get("rating") is not None)
        cell["invalid_rate"] = mean(0 if r["valid"] else 1 for r in rs)
        cell["think_tokens_mean"] = mean(r["think_tokens"] for r in rs)
        cell["total_tokens_mean"] = mean(r["total_tokens"] for r in rs)
        cell["natural_finish_rate"] = mean(1 if r["natural"] else 0 for r in rs) if any(r["think_mode"] for r in rs) else None
        out.setdefault(g, {})[cond] = cell
    return out


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    res = aggregate(rows)
    dst = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace("samples.jsonl", "results_agg.json")
    json.dump(res, open(dst, "w"), indent=1)
    print("wrote", dst, len(res), "groups")


if __name__ == "__main__":
    main()
