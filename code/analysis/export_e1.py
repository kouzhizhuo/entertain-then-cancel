#!/usr/bin/env python3
"""Export finished E1-style runs to the RESULTS_KEYS scheme (CPU only; runs on <host> or locally; skips missing runs).

  python3 analysis/export_e1.py e1   <runs_dir> <out_dir>   # E1_main/<model>  ->  <out_dir>/summary.json (TEST split) + summary_all.json (dev+test)
  python3 analysis/export_e1.py e1s  <runs_dir> <out_dir>   # E1s/<model> (ladder) + E1s_math/<model>/math500.json -> e1s keys
  python3 analysis/export_e1.py a1ctx <runs_dir> <out_dir>  # a1ctx_main/<model> -> R_full, R_partial, KSI
  python3 analysis/export_e1.py e3   <runs_dir> <out_dir>   # E3_gptoss/<model> + E3_check -> effort keys

Key grammar <model>.<set>.<format>.<condition>.<metric>.<stat> (prefix added by json2tex); percentages 0-100; std over seeds, lo/hi 95% bootstrap over ITEMS.
Metrics: R (pragmatic rate) lit dR (vs off) dRb0 (vs b0) acc dacc L tokens natshare rating (1-7 scales, expected rating) ksi Rfull Rpart.
Primary baseline = off; dRb0 is the robustness baseline (lead, 2026-09-30).  'gci' / 'pci' = pooled sets used by the ladder (e1s).
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import numpy as np
HERE = Path(__file__).resolve().parent; EXP = HERE.parent
sys.path.insert(0, str(EXP / "harness")); sys.path.insert(0, str(HERE))
import pilot_summary as PS
import export_keys as EK
for f in sorted((EXP / "data" / "items").glob("[!._]*.jsonl")):
    for l in open(f):
        r = json.loads(l); EK.ITEMS[r["id"]] = r
MODEL = {"Qwen3-0.6B": "qwen3-0p6b", "Qwen3-1.7B": "qwen3-1p7b", "Qwen3-4B": "qwen3-4b", "Qwen3-8B": "qwen3-8b", "Qwen3-14B": "qwen3-14b",
         "Qwen3-32B-FP8": "qwen3-32b", "Qwen3-30B-A3B-FP8": "qwen3-30b-a3b", "gpt-oss-20b": "gpt-oss-20b", "gpt-oss-120b": "gpt-oss-120b",
         "Olmo-3-7B-Think": "olmo3-7b-think", "Olmo-3-32B-Think": "olmo3-32b-think",
         "gpt-5": "gpt-5", "gpt-5.4": "gpt-5p4", "o4-mini": "o4-mini", "claude-sonnet-4.6": "claude-sonnet-4p6", "claude-opus-4.8": "claude-opus-4p8", "gemini-2.5-flash": "gemini-2p5-flash", "gemini-2.5-pro": "gemini-2p5-pro",
         "SmolLM3-3B": "smollm3-3b", "granite-3.3-8b": "granite-3p3-8b", "nemotron-nano-9b-v2": "nemotron-nano-9b-v2", "mimo-7b-sft": "mimo-7b-sft", "mimo-7b-rl": "mimo-7b-rl", "acereason-nemotron-7b": "acereason-7b", "skywork-or1-7b": "skywork-or1-7b", "qwen3-4b-instruct-2507": "qwen3-4b-instruct-2507", "qwen3-4b-thinking-2507": "qwen3-4b-thinking-2507",
         "Qwen3.8-27B": "qwen3p8-27b", "Qwen3.8-27B_low": "qwen3p8-27b-elow", "Qwen3.8-27B_medium": "qwen3p8-27b-emed", "Qwen3.5-0.8B": "qwen3p5-0p8b", "Qwen3.5-2B": "qwen3p5-2b", "Qwen3.5-4B": "qwen3p5-4b", "Qwen3.5-9B": "qwen3p5-9b", "Qwen3.5-35B-A3B": "qwen3p5-35b-a3b",
         "Llama-Nemotron-Nano-8B-v1": "llama-nemotron-nano-8b", "Phi-4-mini-reasoning": "phi4-mini-reasoning", "EXAONE-Deep-7.8B": "exaone-deep-7p8b", "QwQ-32B": "qwq-32b", "Llama-3.3-Nemotron-Super-49B-v1": "llama-nemotron-super-49b",
         "openPangu-Embedded-1B-V1.1": "openpangu-embedded-1b-v1p1", "openPangu-Embedded-7B-V1.1": "openpangu-embedded-7b-v1p1", "openPangu-Embedded-7B-model": "openpangu-embedded-7b-model", "openPangu-Embedded-1B-model": "openpangu-embedded-1b-model", "openPangu-R-7B-2512": "openpangu-r-7b-2512",
         "deepseek_deepseek-v4-flash-0731": "deepseek-v4-flash", "deepseek_deepseek-v4-pro": "deepseek-v4-pro", "DeepSeek-R1-0528-Qwen3-8B": "r1-0528-qwen3-8b",
         "Phi-4": "phi4", "Phi-4-reasoning": "phi4-reasoning", "Phi-4-reasoning-plus": "phi4-reasoning-plus", "DeepSeek-R1-Distill-Llama-8B": "r1distill-llama-8b", "DeepSeek-R1-Distill-Llama-70B": "r1distill-llama-70b",
         "Mistral-Small-3.1-24B-Instruct-2503": "mistral-small-31-instruct", "Magistral-Small-2507": "magistral-small", "Llama-3.1-8B-Instruct": "llama31-8b-instruct",
         "Qwen3-8B-PragReST": "pragrest-8b", "Qwen3-14B-PragReST": "pragrest-14b", "DeepSeek-R1-Distill-Qwen-7B": "r1distill-7b", "DeepSeek-R1-Distill-Qwen-14B": "r1distill-14b"}
TRIG, REL = EK.TRIG, EK.REL
rnd = EK.rnd

def wilson(pct, n, z=1.96):
    """95% Wilson interval (percent) for an accuracy measured on n independent items (single seed: no seed std)."""
    p = pct / 100.0; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return rnd(pct), rnd(100 * (c - w) / d), rnd(100 * (c + w) / d)


def cond_key(c):
    if c in DOSE_RANK: return c
    base, _, hint = c.partition("+")
    t = {"effort_low": "low", "effort_medium": "med", "effort_high": "high", "ist_early": "istE", "ist_late": "istL", "neutral_seed": "neut",
         "off_cot": "cot", "lp4": "lp4", "seed_intent": "sint", "seed_neutral": "sneu", "seed_logic": "slog", "seed_dual": "sdual", "seed_route": "sroute", "seed_task": "stask", "seed_grice": "sgrice", "think_b0_plain": "ntforced"}.get(base) or EK.cond_key(base)
    return t + ({"maxims": "mx"}.get(hint, hint) if hint else "")

def dvs(rows, cond, base, key):
    d = PS.delta(rows, cond, base, key, lambda r: True)
    if d is None: return None
    ps = []
    for s in sorted({r["seed"] for r in rows}):
        rs = [r for r in rows if r["seed"] == s]; a, b = PS.per_item(rs, cond, key), PS.per_item(rs, base, key); ks = set(a) & set(b)
        if ks: ps.append(100 * mean(a[k] - b[k] for k in ks))
    return {"mean": rnd(d[0]), "std": rnd(stdev(ps)) if len(ps) > 1 else None, "lo": rnd(d[1]), "hi": rnd(d[2]), "n": int(d[3])}

def rating_block(rows, cond):
    per, ps = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["condition"] == cond and r.get("rating_exp") is not None: per[r["item_id"]].append(r["rating_exp"]); ps[r["seed"]].append(r["rating_exp"])
    if not per: return None
    v = [mean(x) for x in per.values()]; lo, hi = PS.ci(v)
    return {"mean": rnd(mean(v), 2), "std": rnd(stdev([mean(x) for x in ps.values()]), 2) if len(ps) > 1 else None, "lo": rnd(lo, 2), "hi": rnd(hi, 2), "n": len(v)}

def bucket_sets(r):
    """bucket -> [(set, fmt, kind)] ; kind: 'R' pragmatic-rate set, 'acc' accuracy set, 'rating' 1-7 scale"""
    b, role = r["bucket"], r["role"]
    if b.endswith("_prior"):       # E6 no-utterance probe: same set names with a -prior suffix
        rr = dict(r, bucket=b[:-6]); return [(f"{s}-prior", f, k) for (s, f, k) in bucket_sets(rr)]
    if b in ("a1_fel", "a1_tf", "a1_ter"):
        f = b.split("_")[1]; return [("a1" if role == "target" else "a1-ctrl", f, "R" if role == "target" else "acc")]
    if b == "a1_gra": return [("a1" if role == "target" else "a1-ctrl", "gra", "rating")]
    if b == "a2": return [("a2", "inf", "R"), (f"a2-{r['subset']}", "inf", "R")]
    if b in ("a3_fel", "a3_nli", "a3f_fel", "a3f_nli"):
        f = "inf" if b.endswith("fel") else "nli"; pre = "a3f" if b.startswith("a3f") else "a3"; out = [(pre, f, "R")]   # a3f = full-sample logical-operator sets (E1_scalar_full)
        t = PS_trig(r); rl = r.get("rel")
        if t in TRIG: out.append((f"{pre}-{TRIG[t]}", f, "R"))
        if rl in REL: out.append((f"{pre}r-{REL[rl]}", f, "R"))
        if t in TRIG and rl in REL: out.append((f"{pre}-{TRIG[t]}-{REL[rl]}", f, "R"))   # trigger x relation cells (polarity-balanced analysis)
        return out
    if b == "a4_all_n": return [("a4", "nli", "acc")]
    if b in ("ix_rec", "ix_can"):   # A6 ImplicatureX: accuracy of the truth judgement, per type and pooled
        return [("a6", b[3:], "acc"), (f"a6-{r['subset'].split('_')[0]}", b[3:], "acc")]
    if b == "b3_figqa": return [("b3", "mc", "acc")]
    PX = {"p_a3q": ("quant", "infprior"), "p_a3c": ("conn", "infprior"), "x_a3q": ("quant", "infflip"), "x_a3c": ("conn", "infflip")}   # E1_probe (yes-bias discriminators)
    if b in PX: return [("a3", PX[b][1], "R"), (f"a3-{PX[b][0]}", PX[b][1], "R")]
    if b in ("l_en", "l_de", "l_zh"):                           # non-English slice (E1_xl)
        tg = {"quantifiers": "quant", "connectives": "conn"}.get(r.get("subset"), "x"); return [(f"a3x{b[2:]}", "inf", "R"), (f"a3x{b[2:]}-{tg}", "inf", "R")]
    if b.startswith("m_"): return [(b[2:], "acc", "acc")]                           # METHOD sets (M1_dev / M1_test): set name, fmt token acc
    if b == "c_circa": return [("circa", "mc", "acc")]            # expansion datasets (E1_ds)
    if b == "c_impl": return [("impl1k", "mc", "acc")]
    if b == "c_nope": return [("nope", "nli", "acc")]
    F_ = {"f_a3q": "quant", "f_a3c": "conn", "f_a3m": "modal", "f_a3l": "gradadj"}   # frontier panel buckets
    if b in F_: return [("a3", "inf", "R"), (f"a3-{F_[b]}", "inf", "R")]
    if b == "f_a2": return [("a2", "inf", "R")]
    if b == "f_b2": return [("b2", "yn", "acc")]
    mp_ = re.match(r"^b4_t(\d+)$", b)
    if mp_: return [("b4", "mc", "acc"), (f"b4-t{int(mp_.group(1))}", "mc", "acc")]
    ml_ = re.match(r"^(a3_fel|a2|b2_ludwig)_L(0|[fx]\d+)$", b)
    if ml_:   # E4 load: set a3/a2/b2, format L0 (no load) / Lf512 / Lx2048 ...
        base, code = ml_.groups(); return [({"a3_fel": "a3", "a2": "a2", "b2_ludwig": "b2"}[base], "L" + code, "acc" if base == "b2_ludwig" else "R")]
    mk_ = re.match(r"^b2_k(\d+)$", b)
    if mk_: return [("b2", f"k{mk_.group(1)}", "acc")]
    if b == "ix_prior": return [("a6", "prior", "R"), (f"a6-{r['subset'].split('_')[0]}", "prior", "R")]
    if b.startswith("b1_"): return [(f"b1-{b[3:].replace('_', '')}", "mc", "acc")]
    if b == "b2_ludwig": return [("b2", "yn", "acc")]
    return []

def PS_trig(r): return r.get("trigger")

DOSE_RANK = {"think": 3, "off512": -1, "eff-min": 0, "off": 0, "bud0": 0, "bud128": 1, "eff-low": 1, "bud1k": 2, "eff-med": 2, "bud4k": 3, "eff-high": 3, "bud8k": 3, "bud16k": 4}


def base_of(conds):
    if conds and all(c in DOSE_RANK for c in conds): return min(conds, key=lambda c: DOSE_RANK[c])   # frontier panel: delta vs the smallest dose
    """Baseline condition of a run: gpt-oss -> effort_low ('low'); OLMo-3-Think (no off arm) -> think_b0_plain ('ntforced'); Qwen3 -> off."""
    if any(c.startswith("effort_") for c in conds): return "effort_low"
    if "think_b0_plain" in conds and "off" not in conds: return "think_b0_plain"
    return "off"


def export_model(R, split):
    rows = [r for r in R["rows"] if split is None or r.get("split") == split]
    free = [r for r in R["free"] if split is None or r.get("split") == split]
    conds = [c for c in R["meta"]["conditions"]]
    base = base_of(conds)
    if base == "think_b0_plain" and "think_b512_plain" in conds:   # always-thinking models can answer in the reopened thought after a forced empty </think>: if the NT-forced arm is unusable, the baseline becomes the smallest valid cap
        bv = [r["valid"] for r in rows if r["condition"] == "think_b0_plain"]
        if bv and sum(bv) / len(bv) < 0.05: base = "think_b512_plain"   # only when the NT-forced arm is essentially unusable (<5% valid): Qwen3-4B-Thinking-2507; Phi-4-reasoning-plus (34% valid) keeps ntforced
    by = defaultdict(list)
    for r in rows:
        for s, f, k in bucket_sets(r): by[(s, f, k)].append(r)
    out = {}
    for (s, f, kind), srows in by.items():
        nf = out.setdefault(s, {}).setdefault(f, {})
        for c in conds:
            node = nf.setdefault(cond_key(c), {})
            if kind == "rating":
                node["rating"] = rating_block(srows, c)
            else:
                key = "R" if kind == "R" else "acc"
                node[key] = EK.rate_block(srows, c, "is_gold")
                if s.startswith("b1-"): node["lit"] = EK.rate_block(srows, c, "is_literal")
                elif kind == "R": node["lit"] = EK.rate_block(srows, c, "is_literal")
                if c != base:
                    node["d" + key] = dvs(srows, c, base, "is_gold")
                    if "think_b0" in conds and c != "think_b0": node["d" + key + "b0"] = dvs(srows, c, "think_b0", "is_gold")
                if "think_b0_plain" in conds and base != "think_b0_plain" and c != "think_b0_plain":
                    node["d" + key + "nt"] = dvs(srows, c, "think_b0_plain", "is_gold")   # delta vs the NT-forced baseline (comparable with OLMo)
                if f not in ("ter",) and s not in ("a4",) or True:
                    try: node["L"] = EK.L_block(srows, c)
                    except Exception: pass
            tok, nat = EK.token_block(srows, c); node["tokens"] = tok
            if nat: node["natshare"] = nat
            for k in [k for k, v in node.items() if v is None]: del node[k]
        for c in ("off", "think_full"):
            if any(r["condition"] == c for r in free):
                for k, v in EK.free_blocks(free, None, c).items(): pass
    EK.add_digits(out); return out

def pooled(R, split, name, specs):
    base = base_of(R["meta"]["conditions"])
    """pooled sets for the ladder: specs = [(bucket, role)] -> set `name`, fmt 'fel' (GCI) or 'nat' accuracy (PCI)"""
    rows = [r for r in R["rows"] if (split is None or r.get("split") == split) and (r["bucket"], r["role"]) in specs]
    out = {}
    for c in R["meta"]["conditions"]:
        node = out.setdefault(cond_key(c), {}); key = "R" if name == "gci" else "acc"
        node[key] = EK.rate_block(rows, c, "is_gold")
        if c != base: node["d" + key] = dvs(rows, c, base, "is_gold")
        if "think_b0_plain" in R["meta"]["conditions"] and base != "think_b0_plain" and c != "think_b0_plain":
            node["d" + key + "nt"] = dvs(rows, c, "think_b0_plain", "is_gold")
        tok, nat = EK.token_block(rows, c); node["tokens"] = tok
        for k in [k for k, v in node.items() if v is None]: del node[k]
    return out

def ksi(R, split, frame):
    rows = [r for r in R["rows"] if (split is None or r.get("split") == split)]
    fb = {"felicity": "a1ctx_fel", "inference": "a1ctx_inf"}[frame]; out = {}
    for c in R["meta"]["conditions"]:
        full, part = defaultdict(list), defaultdict(list)
        for r in rows:
            if r["bucket"] != fb or r["condition"] != c or r["is_gold"] is None: continue
            sc = int(r["item_id"].split(".")[1])
            (full if r["role"] == "target" else part)[sc].append(float(r["is_gold"]))
        ks = sorted(set(full) & set(part))
        if not ks: continue
        rf = [mean(full[k]) for k in ks]; rp = [1 - mean(part[k]) for k in ks]; d = [a - b for a, b in zip(rf, rp)]
        def blk(v):
            lo, hi = PS.ci(v); return {"mean": rnd(100 * mean(v)), "lo": rnd(100 * lo), "hi": rnd(100 * hi), "n": len(v)}
        out[cond_key(c)] = {"Rfull": blk(rf), "Rpart": blk(rp), "ksi": blk(d)}
    return out

def _effort(ck):
    out = {{"low": "low", "medium": "med", "high": "high"}[k]: v for k, v in ck.items() if k in ("low", "medium", "high")}
    out["monotone"] = 1 if ck.get("monotone_low_lt_medium_lt_high") else 0
    return out


def deep_new_only(dst, src):
    """add keys of src that dst lacks (never overwrite): lets an extra run contribute only NEW condition nodes / delta keys to a model's tree."""
    for k, v in src.items():
        if k not in dst: dst[k] = v
        elif isinstance(v, dict) and isinstance(dst[k], dict): deep_new_only(dst[k], v)


def e1s_node(R, sp):
    node = {"gci": {"fel": pooled(R, sp, "gci", {("a1_fel", "target"), ("a3_fel", "target"), ("a2", "target")})},
            "pci": {"nat": pooled(R, sp, "pci", {("b2_ludwig", "target"), ("b1_indirect_speech", "target"), ("b1_irony", "target")})}}
    node.update({k: v for k, v in export_model(R, sp).items()})
    return node


def main():
    mode, runs, outd = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); outd.mkdir(parents=True, exist_ok=True)
    res = {"test": {}, "all": {}}
    sub = {"e1r1": "E1R1_core", "e1": "E1_main", "e1s": "E1s", "a1ctx": "a1ctx_main", "e3": "E3_gptoss", "e8": "E8", "e6": "E6", "e2": "E2_lp", "e2t": "E2_think", "e2c": "E2_cot", "e1sb": "E1s_b0", "e1x": "E1_seeds2", "e3x": "E3_gptoss_s2", "e7": "E7", "a6": "A6_ix", "e9": "E9_ruis", "e9h": "E9_hu_lp", "e4": "E4_load", "e1xt": "E1_extra", "e1f": "E1_scalar_full", "e1pci": "E1_pci_big", "e1p": "E1p_pop", "e1fr": "E1f", "e1b": "E1b", "e1q": "E1q", "e9g": "E9_grice", "e7f": "E7_full", "e1probe": "E1_probe", "e1ds": "E1_ds", "e10": "E10_seeds", "e1dsx": "E1_dsx", "e1xl": "E1_xl", "e1sfx": "E1_sfx", "e3ds": "E1_gptoss_ds", "e3sf": "E1_gptoss_sf", "e1pangu": "E1_pangu", "m1dev": "M1_dev", "m1test": "M1_test"}[mode]
    for mdir in sorted((runs / sub).glob("*")) if (runs / sub).exists() else []:
        if not (mdir / "results.json").exists(): continue
        R = PS.load(mdir); mk = MODEL.get(mdir.name, mdir.name.lower().replace('.', 'p'))
        for sp, key in (("test", "test"), (None, "all")):
            if mode in ("e1", "e3", "e8", "e6", "e2", "e2t", "e2c", "e1sb", "e1x", "e3x", "e7", "a6", "e9", "e9h", "e4", "e1xt", "e1f", "e1pci", "e1p", "e1fr", "e1b", "e1q", "e9g", "e7f", "e1probe", "e1ds", "e10", "e1dsx", "e1xl", "e1sfx", "e3ds", "e3sf", "e1pangu", "m1dev", "m1test", "e1r1"): node = export_model(R, sp)
            elif mode == "a1ctx": node = {"a1ctx-fel": {"fel": ksi(R, sp, "felicity")}, "a1ctx-inf": {"inf": ksi(R, sp, "inference")}}
            else:
                node = e1s_node(R, sp)
            res[key][mk] = node
        print("exported", sub, mdir.name, flush=True)
    if mode == "e1r1":      # block E1R1 = core (E1R1_core) + 3-way full sets (E1R1_sfx) for DeepSeek-R1-0528-Qwen3-8B, same node shapes as e1p / e1sfx
        for mdir in sorted((runs / "E1R1_sfx").glob("*")) if (runs / "E1R1_sfx").exists() else []:
            if not (mdir / "results.json").exists(): continue
            R = PS.load(mdir); mk = MODEL.get(mdir.name, mdir.name.lower().replace(".", "p"))
            for sp, key in (("test", "test"), (None, "all")):
                deep_new_only(res[key].setdefault(mk, {}), export_model(R, sp))
            print("merged E1R1_sfx", mdir.name, flush=True)
    if mode == "e1s":
        for extra in ("E1s_nt", "E1s_b0"):
            if not (runs / extra).exists(): continue
            for mdir in sorted((runs / extra).glob("*")):
                if not (mdir / "results.json").exists(): continue
                R = PS.load(mdir); mk = MODEL.get(mdir.name, mdir.name.lower().replace(".", "p"))
                for sp, key in (("test", "test"), (None, "all")):
                    deep_new_only(res[key].setdefault(mk, {}), e1s_node(R, sp))
                print(f"merged (new keys only) {extra}", mdir.name, flush=True)
    if mode == "e1s":
        for md in sorted((runs / "E1s_math").glob("*")) if (runs / "E1s_math").exists() else []:
            f = md / "math500.json"
            if not f.exists(): continue
            m = json.load(open(f))["results"]; mk = MODEL.get(md.name, md.name.lower().replace('.', 'p'))
            for key in ("test", "all"):
                cap = res[key].setdefault(mk, {}).setdefault("cap", {}).setdefault("math500", {})
                for cond, v in m.items():
                    res[key][mk].setdefault("math500", {})["off" if cond == "nt" else cond] = {"acc": dict(zip(("mean", "lo", "hi"), wilson(v["acc"], v["n"])), n=v["n"]), "tokens": {"mean": rnd(v["tokens_mean"], 0)}, "unfinished": {"mean": rnd(v["unfinished_share"])}}
                    cap[cond] = {"acc": dict(zip(("mean", "lo", "hi"), wilson(v["acc"], v["n"])), n=v["n"]), "tokens": {"mean": rnd(v["tokens_mean"], 0)}, "unfinished": {"mean": rnd(v["unfinished_share"])}}
    if mode == "e3" and (runs / "E3_check").exists():
        for d in (runs / "E3_check").glob("*/effort_check.json"):
            for key in ("test", "all"): res[key].setdefault(MODEL.get(d.parent.name, d.parent.name.lower().replace(".", "p")), {})["effort_check"] = _effort(json.load(open(d)))
    json.dump(res["test"], open(outd / "summary.json", "w"), indent=1)
    json.dump(res["all"], open(outd / "summary_all.json", "w"), indent=1)
    json.dump({"mode": mode, "split": "test (summary.json) / dev+test (summary_all.json)", "models": sorted(res["test"])}, open(outd / "DONE", "w"))
    print("wrote", outd / "summary.json")

if __name__ == "__main__":
    main()
