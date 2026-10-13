#!/usr/bin/env python3
"""Print a compact digest of the newest P2 blocks after a merge (E7/E7F trajectories + markers for new models, E10 seeds, E1PROBE, E1XL)."""
import json, sys
from pathlib import Path
X = Path(__file__).resolve().parents[1] / "results"
def load(n):
    f = X / n / "summary.json"; return json.load(open(f)) if f.exists() else {}
for blk, sets in (("E7", ("a3-quant", "a3-conn")), ("E7F", ("a3f-quant", "a3f-conn"))):
    d = load(blk)
    for m in sorted(d):
        if m.startswith("qwen3-") and blk == "E7": continue
        for s in sets:
            n = d[m].get(s, {}).get("inf") or {}
            if "traj" not in n: continue
            t = n["traj"]; k = n.get("marker") or {}
            print(blk, m, s, f"n={n['traj_n']['traces']} drop={n['traj_n']['dropped_invalid']} a/b/c/d/e=" + "/".join(f"{t[c]['share']:.0f}" for c in "abcde"),
                  ("marker pre/post/none=%.0f/%.0f/%.0f ctrl=%.0f" % (k["pre"]["mean"], k["post"]["mean"], k["none"]["mean"], k["ctrl_marker"]["mean"])) if k.get("pre") else "", "alive512=%.0f" % n["alive"]["k0512"]["share"])
d = load("E10")
for m in sorted(d):
    for s in ("a3-quant", "a3-conn", "a2"):
        n = (d[m].get(s) or {}); n = n[list(n)[0]] if n else {}
        r = [f"{c}={n[c]['R']['mean']:.1f}" for c in ("off", "nat", "sint", "sneu", "slog") if c in n and "R" in n[c]]
        if r: print("E10", m, s, " ".join(r))
d = load("E1PROBE")
for m in sorted(d):
    for f in ("infprior", "infflip"):
        n = (d[m].get("a3") or {}).get(f)
        if n and "off" in n and "nat" in n:
            dd = n["nat"].get("dR") or {}; print("PROBE", m, f, "off R %.1f nat R %.1f" % (n["off"]["R"]["mean"], n["nat"]["R"]["mean"]), "dR %+.1f [%+.1f,%+.1f]" % (dd.get("mean", 0), dd.get("lo", 0), dd.get("hi", 0)))
d = load("E1XL")
for m in sorted(d):
    for s in sorted(d[m]):
        if not s.startswith("a3x"): continue
        n = d[m][s].get("inf") or {}; c = "off" if "off" in n else None
        if c and "nat" in n and "dR" in n["nat"]: print("XL", m, s, "off %.1f nat %.1f dR %+.1f [%+.1f,%+.1f] n=%d" % (n[c]["R"]["mean"], n["nat"]["R"]["mean"], n["nat"]["dR"]["mean"], n["nat"]["dR"]["lo"], n["nat"]["dR"]["hi"], n["nat"]["dR"]["n"]))
