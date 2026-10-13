#!/usr/bin/env python3
"""RUN MANIFEST (lead request): exp/results/MANIFEST/runs.json, one entry per block the paper uses. For every raw run directory of the block: model, config name and sha, item set (n items), seeds, conditions,
sampling (temperature / top_p / top_k, thinking cap), stop rule, number of forced-choice samples, invalid answers (no readable option) and stage-1 finish counts (thoughts that hit the cap), host/GPU/vllm version and the code hashes. Reads results/<src>/<group>/<model>/{results.json,samples.jsonl}."""
import json, re, sys
from collections import Counter
from pathlib import Path
X = Path(__file__).resolve().parents[1] / "results"; SRC = ("runs", "<host>", "slurm1", "slurm2", "slurm3", "slurm4")
BLOCKS = {"e1": ["E1_main"], "e1s": ["E1s", "E1s_nt", "E1s_b0", "E1s_s2"], "e1sf": ["E1_scalar_full"], "e1sfx": ["E1_sfx"], "e1q": ["E1q", "E1q_s2"], "e1p": ["E1p_pop"], "e1b": ["E1b", "E1b_s2"], "e1pangu": ["E1_pangu"],
          "e1r1": ["E1R1_core", "E1R1_sfx"], "e3": ["E3_gptoss", "E3_gptoss_s2"], "e1gsf": ["E1_gptoss_sf"], "e1probe": ["E1_probe"], "e1ds": ["E1_ds", "E1_dsx", "E1_gptoss_ds"], "e1xl": ["E1_xl"], "e10": ["E10_seeds"],
          "e7": ["E7"], "e7f": ["E7_full"], "e2": ["E2_lp"], "e2t": ["E2_think"], "e2c": ["E2_cot"], "e6": ["E6"], "e8": ["E8"], "a1ctx": ["a1ctx_main"], "a6": ["A6_ix"], "e9": ["E9_ruis", "E9_hu_lp", "E9_grice"],
          "m1dev": ["M1_dev", "M1_dev_v2", "M1_dev_D", "M1_dev_X", "M1_dev_SC"], "m1dev_gsm": ["M1_dev_v2_gsm", "D_dev_gsm", "M1_dev_gsm"]}
out = {}
for blk, groups in BLOCKS.items():
    runs = []
    for g in groups:
        for src in SRC:
            for d in sorted((X / src / g).glob("*")) if (X / src / g).exists() else []:
                f = d / "results.json"
                if not f.exists() or d.name.startswith(("BADTOK", "bad_")): continue
                try: r = json.load(open(f))
                except Exception: continue
                m = r.get("meta", r); ent = {"group": g, "source": src, "run": d.name, "model": m.get("model_name"), "config": (m.get("config_file") or "").split("/")[-1], "config_sha16": m.get("config_sha16")}
                for k in ("seeds", "conditions", "n_items", "n_samples", "sampling", "host", "gpu", "vllm", "finished"): ent[k] = m.get(k)
                ent["code_sha16"] = m.get("code_sha16"); ent["stage1_finish_counts"] = m.get("finish_counts")
                if isinstance(m.get("config"), str):
                    mm = re.search(r"'description': (.{0,300})", m["config"]); ent["config_description"] = mm.group(1)[:300] if mm else None
                sf = d / "samples.jsonl"
                if sf.exists():
                    n = bad = 0; cnt = Counter()
                    for l in open(sf):
                        try: row = json.loads(l)
                        except Exception: continue
                        n += 1; bad += (row.get("valid") is False)
                        if "stage1_finish" in row and row.get("think_mode"): cnt[row["stage1_finish"]] += 1
                    ent["forced_choice_rows"] = n; ent["invalid_answers"] = bad; ent["invalid_share_pct"] = round(100 * bad / max(1, n), 3); ent["think_rows_by_stage1_finish"] = dict(cnt)
                elif "results" in r: ent["gsm_arms"] = {c: {k: v.get(k) for k in ("n", "acc", "tokens_mean", "unfinished_share")} for c, v in r["results"].items()}
                runs.append(ent)
    if runs: out[blk] = {"groups": groups, "n_runs": len(runs), "models": sorted({e["model"] or e["run"] for e in runs}), "stop_rule": "stage 1 stops at the close-of-thinking delimiter or the cap; stage 2 forces the close and reads the option-letter logprobs greedily (letter-restricted argmax)", "runs": runs}
# API block (closed models + DeepSeek-V4 via the self-hosted endpoints): dir listing and call counts from the finalize log
api = X / "api"; ent = []
if api.exists():
    for d in sorted(api.glob("*")):
        if d.is_dir(): ent.append({"model_dir": d.name, "arms": sorted(p.name for p in d.glob("*") if p.is_dir())})
    out["e1f"] = {"groups": ["api", "api_mc", "api_flip", "api_prior"], "n_runs": len(ent), "models": [e["model_dir"] for e in ent], "runs": ent, "note": "API runs: reasoning effort / thinking toggle per arm, 5 seeds on the item sets in configs/E1f_*.json; call counts and cost in results/api_logs/finalize*.log; no new closed-model runs after 2026-10-07 (user rule)."}
(X / "MANIFEST").mkdir(exist_ok=True); json.dump(out, open(X / "MANIFEST/runs.json", "w"), indent=1)
print({b: v["n_runs"] for b, v in out.items()})
