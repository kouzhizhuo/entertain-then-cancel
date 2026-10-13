#!/usr/bin/env python3
"""Inject the E7 marker-onset blocks (results/runs/E7/<model>/marker.json from harness/e7_marker_onset.py, computed where traces.jsonl lives) into results/E7/summary.json
as e7.<model>.<set>.inf.marker.{pre,post,none,ctrl_marker}.{mean,lo,hi,n} (+ n_turn_traces, n_traces). Called by merge_results.sh after the E7 export."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_e1 import MODEL
X = Path(__file__).resolve().parents[1] / "results"
for grp, dst in (("E7", "E7"), ("E7_full", "E7F")):
  S = X / dst / "summary.json"
  if not S.exists(): continue
  sm = json.load(open(S)); n = 0
  for fn, key in (("marker.json", None), ("traj.json", "traj")):
    for f in sorted(g for src in ("runs", "<host>", "slurm4") for g in (X / src / grp).glob("*/" + fn)):
      mk = MODEL.get(f.parent.name, f.parent.name.lower().replace(".", "p"))
      for s, node in json.load(open(f)).items():
          tgt = sm.setdefault(mk, {}).setdefault(s, {}).setdefault("inf", {})
          if key is None: tgt["marker"] = node
          else: tgt.update({"traj": node["traj"], "traj_k32": node["traj_k32"], "traj512": node["traj512"], "alive": node["alive"], "traj_n": {"traces": node["n_traces"], "items": node["n_items"], "dropped_invalid": node["n_dropped_invalid"], "alive512_traces": node["n_traces_alive512"]}})
          n += 1
  json.dump(sm, open(S, "w"), indent=1); print("e7 marker blocks merged:", n)
