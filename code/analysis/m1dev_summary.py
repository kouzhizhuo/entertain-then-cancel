"""Export the M1 development selection files (v1 seeds, v2 late check) as a json2tex summary.
Keys: m1dev.qwen3-8b.<v1|v2>.<arm>.<speaker|literal|presup|score|tokfc|tokgsm>.mean, and
m1dev.qwen3-8b.v2.<arm>.tr.<family|set>.<w2r|r2w|n>.mean (paired transitions vs NAT)."""
import json, os
R = os.path.join(os.path.dirname(__file__), "..", "results")
ARM = {"off": "nt", "think_full": "nat", "think_b512_plain": "nat512", "think_full+taskaware": "taskaware",
       "think_full+sitroute": "userroute", "seed_intent": "intent", "seed_dual": "dual", "seed_route": "route",
       "seed_task": "taskcheck", "seed_grice": "grice", "late_sitcheck": "sitcheck", "late_neutral": "neutral", "nat": "nat", "sit": "sit", "base_NAT": "nat", "SIT-train_selected": "sit", "control_unselected": "control", "SIT-nobal_ablation": "nobal", "dpo": "dpo", "dpo_shuffled": "dposhuf", "x1": "x1", "x1_w0": "x1w0", "x2": "x2", "x2_w0": "x2w0", "control": "control", "nobal": "nobal"}
MET = {"speaker": "speaker", "literal": "literal", "presup": "presup", "score": "score",
       "tokens_forced_choice": "tokfc", "tokens_gsm": "tokgsm"}
out = {}
for ver, f in (("v1", "selection_dev_Qwen3-8B.json"), ("v2", "selection_dev_v2_Qwen3-8B.json"), ("v3", "selection_dev_v3_Qwen3-8B.json"),
               ("d", "selection_dev_D_q3.json"), ("d-r1", "selection_dev_D_r1.json"), ("e", "selection_dev_E_q3.json"), ("e-r1", "selection_dev_E_r1.json"), ("x", "selection_dev_X_q3.json"), ("x-r1", "selection_dev_X_r1.json")):
    if not os.path.exists(os.path.join(R, "M1", f)): continue
    d = json.load(open(os.path.join(R, "M1", f)))
    mdl = "r1-0528" if ver.endswith("-r1") else "qwen3-8b"; ver = ver.replace("-r1", "")
    blk = out.setdefault(mdl, {}).setdefault(ver, {})
    if "routed_total" in d: blk["routed"] = {"mean": d["routed_total"]}
    for arm, row in d["table"].items():
        blk[ARM.get(arm, arm)] = {MET[k]: {"mean": round(v, 3)} for k, v in row.items() if k in MET}
        blk[ARM.get(arm, arm)]["gates"] = {"mean": int(bool(row.get("gates_pass")))}
    for arm, tr in d.get("transitions_vs_nat", {}).items():
        t = blk.setdefault(ARM.get(arm, arm), {}).setdefault("tr", {})
        for grp in ("family", "sets"):
            for name, c in tr[grp].items():
                t[name.replace("_", "-")] = {"w2r": {"mean": c["wrong_to_right"]}, "r2w": {"mean": c["right_to_wrong"]}, "n": {"mean": c["n"]}}
os.makedirs(os.path.join(R, "M1DEV"), exist_ok=True)
json.dump(out, open(os.path.join(R, "M1DEV", "summary.json"), "w"), indent=1)
print("wrote M1DEV/summary.json")
