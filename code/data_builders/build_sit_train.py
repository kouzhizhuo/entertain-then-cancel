#!/usr/bin/env python3
"""D-CONTINGENCY (SIT-train) TRAIN POOL. Writes exp/splits/sit_train_families.json (frozen BEFORE any sampling, sha256 inside), data/items/sit_train.jsonl (forced choice) and data/items/sit_train_gsm.jsonl.
Never reads METHOD TEST outcomes; only item ids/texts of dev/test/inspected items are read, to EXCLUDE them. Run: python3 build_sit_train.py [inspected_ids.json]

Pool rules (notes/METHOD_DESIGN.md D-CONTINGENCY; every deviation is written into the split file under "family_disjointness"):
  speaker:  IMPPRES felicity framing, triggers gradable_adjective / gradable_verb / numerals_10_100 / numerals_2_3 (trigger families never used in DEV/TEST), uninspected ids only;
            Circa (train.parquet rows not in DEV/TEST and with no Y-answer text used there); PUB task 4 (implicature recovery) and task 5 (agreement detection) -- task-disjoint from the eval tasks 2/3/6;
            Implicatures1k rows never inspected (ImplicatureX is fully inspected/used, so none of it is available).
  presup:   IMPPRES presupposition items not in DEV/TEST/inspected and whose premise sentence is not used there; PUB task 11 (presupposition NLI; premise-sentence overlap with eval IMPPRES presup removed) and task 12 (presupposition over QA).
            NOPE: all 692 items are in DEV/TEST -> none available.
  literal:  IMPPRES truth framing of the SAME sentence pairs as the felicity items (gold = IMPPRES logical gold); FOLIO train minus the 200 dev items and minus their stories; GSM8K train minus the 150 dev items.
  LUDWIG: the release has only dev/test, all in DEV/TEST/inspected -> none available.  ImplicatureX can/rec: all inspected -> none available."""
import hashlib, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; RAW, ITEMS = HERE / "raw", HERE / "items"; SPL = HERE.parent / "splits"; SPL.mkdir(exist_ok=True)
L = lambda n: [json.loads(l) for l in open(ITEMS / f"{n}.jsonl")]
insp = set(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/inspected_ids.json")))
dev = json.load(open(HERE / "manifests" / "METHOD_DEV.json"))["sets"]; test = json.load(open(HERE / "manifests" / "METHOD_TEST.json"))["sets"]
evalids = {i for v in list(dev.values()) + list(test.values()) for i in v}
def orig(s): return {i.split(".", 2)[2] for i in dev.get(s, []) + test.get(s, [])}
norm = lambda s: " ".join(str(s).lower().split())
CAP = {"imppres": 600, "circa": 600, "pub_task4": 300, "pub_task5": 300, "impl1k": 500, "imppres_presup": 900, "pub_task11": 700, "pub_task12": 700, "folio": 700, "gsm8k": 700}
out, why = [], {}
def add(setname, family, base, ctx, q, opts, kinds, gold, oid, extra=None):
    it = {k: base.get(k) for k in ("phenomenon", "source", "subset")}
    it.update(id=f"sit.{setname}.{oid}", role="target", context=ctx, question=q, options=opts, option_kinds=kinds, gold=gold, literal=None, question_alt=None, variants={}, split="train",
              meta={**base.get("meta", {}), "set": setname, "family": family, "orig_id": oid, **(extra or {})})
    out.append(it)

# ---- IMPPRES felicity (speaker) + truth (literal) on trigger families absent from DEV/TEST
sc = L("imppres_scalar_target"); used_orig = orig("imppres_quant_fel") | orig("imppres_conn_fel")
avail = defaultdict(list)
for b in sc:
    if b["subset"] in ("quantifiers", "connectives", "modals") or b["id"] in insp or b["id"] in used_orig: continue
    avail[b["subset"]].append(b)
rng = random.Random(21); sel = []
for t in sorted(avail): rows = sorted(avail[t], key=lambda x: x["id"]); rng.shuffle(rows); sel += rows[: CAP["imppres"] // len(avail)]
why["imppres_triggers_used"] = {t: len(v) for t, v in avail.items()}
for b in sorted(sel, key=lambda x: x["id"]):
    fel = b["variants"]["felicity"]
    add("imppres_fel", "speaker", b, fel["context"], fel["question"], fel["options"], fel["option_kinds"], fel["gold"], b["id"], {"trigger": b["subset"]})
    gy = 0 if str(b["meta"]["gold_label_log"]) == "0" else 1
    add("imppres_truth", "literal", b, fel["context"], "Does the second statement strictly follow, by logic alone, from the first? Answer Yes or No.", ["Yes", "No"], ["entails", "does_not_entail"], gy, b["id"], {"trigger": b["subset"]})

# ---- Circa (speaker): same filter as circa_items, rows disjoint from DEV/TEST (index and Y-answer text)
sys.path.insert(0, str(HERE)); import importlib.util
src = open(HERE / "build_items.py").read()
DIRECT = set(eval(src[src.index("DIRECT = ") + 9: src.index("\n", src.index("DIRECT = "))]))
d = pd.read_parquet(RAW / "circa" / "train.parquet"); d = d[d["goldstandard2"].isin([0, 1, 2])].copy()
toks = d["answer-Y"].str.lower().str.findall(r"[a-z']+"); d = d[~toks.apply(lambda t: any(w in DIRECT for w in t))].reset_index(drop=True)
used_c = {int(x.split("m1.circa.circa.")[1]) for x in dev["circa"] + test["circa"]}
used_ans = {norm(d.loc[i, "answer-Y"]) for i in used_c if i in d.index}; used_q = {norm(d.loc[i, "question-X"]) for i in used_c if i in d.index}
cn = 0
for g, k, nm in ((0, 240, "yes"), (1, 240, "no"), (2, 120, "middle")):
    sub = d[(d["goldstandard2"] == g) & ~d.index.isin(used_c) & ~d["answer-Y"].map(norm).isin(used_ans)]; idx = list(sub.index); random.Random(30 + g).shuffle(idx)
    for i in sorted(idx[:k]):
        r = d.loc[i]; cn += 1
        add("circa", "speaker", {"phenomenon": "indirect_answer", "source": "circa", "subset": "indirect"}, f"Context: {r['context']}\nX asks: \"{r['question-X']}\"\nY answers: \"{r['answer-Y']}\"", "What does Y's answer mean?",
            ["Yes", "No", "Neither yes nor no (in the middle)"], ["yes", "no", "middle"], g, f"{i}", {"gold_class": nm})
why["circa_excluded_rows"] = len(used_c)

# ---- PUB (task-disjoint from eval tasks 2/3/6)
p6ctx = {norm(b["context"]) for b in L("pub6")}
ev_presup_text = set()
for s in ("imppres_presup",):
    for b in L("imppres_presupposition"):
        if b["id"] in orig(s):
            c = b["context"].split("\n"); ev_presup_text |= {norm(c[0].replace("First sentence:", "")), norm(c[1].replace("Second sentence:", "")) if len(c) > 1 else ""}
for t, fam, key in ((4, "speaker", "pub_task4"), (5, "speaker", "pub_task5"), (11, "presup", "pub_task11"), (12, "presup", "pub_task12")):
    rows = [b for b in L(f"pub_task{t:02d}") if norm(b["context"]) not in p6ctx and b["id"] not in insp]
    if t == 11:
        rows = [b for b in rows if not any(norm(x.split(":", 1)[1]) in ev_presup_text for x in b["context"].split("\n") if ":" in x)]
    random.Random(40 + t).shuffle(rows)
    for b in sorted(rows[: CAP[key]], key=lambda x: x["id"]):
        add(key, fam, b, b["context"], b["question"], b["options"], b["option_kinds"], b["gold"], b["id"])
    why[f"{key}_available"] = len(rows)

# ---- Implicatures1k remainder (speaker)
i1 = [b for b in L("implicatures1k") if b["id"] not in insp]; random.Random(50).shuffle(i1); why["impl1k_available"] = len(i1)
for b in sorted(i1[: CAP["impl1k"]], key=lambda x: x["id"]): add("impl1k", "speaker", b, b["context"], b["question"], b["options"], b["option_kinds"], b["gold"], b["id"])

# ---- IMPPRES presupposition (presup): item-disjoint and premise-sentence-disjoint; 99/100 paradigms are shared with DEV/TEST (NOT family-disjoint)
pp = [b for b in L("imppres_presupposition") if b["id"] not in insp and b["id"] not in orig("imppres_presup")]
pp = [b for b in pp if norm(b["context"].split("\n")[0].replace("First sentence:", "")) not in ev_presup_text]; why["imppres_presup_available"] = len(pp)
random.Random(60).shuffle(pp)
for b in sorted(pp[: CAP["imppres_presup"]], key=lambda x: x["id"]): add("imppres_presup", "presup", b, b["context"], b["question"], b["options"], b["option_kinds"], b["gold"], b["id"])

# ---- FOLIO train (literal): drop the 200 dev items and every story that has a dev item
fo = [json.loads(l) for l in open(RAW / "folio" / "folio_v2_train.jsonl")]; rf = random.Random(7); tr = list(fo); rf.shuffle(tr)
devrows = sorted(tr[:200], key=lambda r: str(r["example_id"])); dev_ids = {str(r["example_id"]) for r in devrows}; dev_st = {str(r["story_id"]) for r in devrows}
rest = [r for r in fo if str(r["example_id"]) not in dev_ids and str(r["story_id"]) not in dev_st]; why["folio_available"] = len(rest); random.Random(70).shuffle(rest)
LAB = {"True": 0, "False": 1, "Uncertain": 2}
for r in sorted(rest[: CAP["folio"]], key=lambda r: str(r["example_id"])):
    add("folio", "literal", {"phenomenon": "logic", "source": "folio", "subset": "train", "meta": {"story_id": str(r["story_id"])}}, f"Premises:\n{r['premises'].strip()}\n\nConclusion: {r['conclusion'].strip()}",
        "Based only on the premises, is the conclusion true, false, or uncertain?", ["True", "False", "Uncertain"], ["true", "false", "uncertain"], LAB[r["label"]], str(r["example_id"]))

# ---- GSM8K train (literal, free numeric)
g = pd.read_parquet(RAW / "gsm8k" / "train.parquet"); idx = list(range(len(g))); random.Random(11).shuffle(idx); devg = set(idx[:150])
rest_g = [i for i in range(len(g)) if i not in devg]; random.Random(80).shuffle(rest_g); gs = []
for i in sorted(rest_g[: CAP["gsm8k"]]):
    gs.append({"id": f"sit.gsm8k.{i}", "problem": g.iloc[i].question, "answer": g.iloc[i].answer.split("####")[-1].strip().replace(",", ""), "split": "train", "meta": {"set": "gsm8k", "family": "literal", "orig_id": str(i)}})

# ---- hard disjointness checks, then freeze
assert not ({it["meta"]["orig_id"] for it in out if it["meta"]["set"] in ("circa",)} & {x.split("m1.circa.circa.")[1] for x in dev["circa"] + test["circa"]})
assert not {it["meta"]["orig_id"] for it in out if it["meta"]["set"] == "folio"} & ({i.split(".", 2)[2] for i in dev["folio"] + test["folio"]})
assert not {gi["meta"]["orig_id"] for gi in gs} & {i.split("dev")[-1] for i in dev["gsm8k"]}
assert not {it["meta"]["orig_id"] for it in out if it["meta"]["set"] in ("imppres_fel", "imppres_truth")} & (orig("imppres_quant_fel") | orig("imppres_conn_fel") | insp)
with open(ITEMS / "sit_train.jsonl", "w") as f:
    for it in out: f.write(json.dumps(it, ensure_ascii=False) + "\n")
with open(ITEMS / "sit_train_gsm.jsonl", "w") as f:
    for r in gs: f.write(json.dumps(r) + "\n")
sets = defaultdict(list)
for it in out: sets[it["meta"]["set"]].append(it["id"])
sets["gsm8k"] = [r["id"] for r in gs]
fam = {"speaker": ["imppres_fel", "circa", "pub_task4", "pub_task5", "impl1k"], "presup": ["imppres_presup", "pub_task11", "pub_task12"], "literal": ["imppres_truth", "folio", "gsm8k"]}
spl = {"frozen": "2026-10-05 (before any D sampling; no test access)", "families": fam, "sets": {k: sorted(v) for k, v in sets.items()},
       "pool_sizes": {k: len(v) for k, v in sets.items()}, "family_pool_sizes": {f: sum(len(sets[s]) for s in ss) for f, ss in fam.items()}, "availability": why, "caps": CAP,
       "excluded": {"inspected_item_ids_in_earlier_runs": len(insp), "dev_items": sum(len(v) for v in dev.values()), "test_items": sum(len(v) for v in test.values())},
       "family_disjointness": {
           "imppres_fel / imppres_truth": "TRIGGER-family disjoint from DEV/TEST (gradable_adjective, gradable_verb, numerals_10_100, numerals_2_3; DEV/TEST use quantifiers and connectives); modals fully inspected -> excluded; item-disjoint from every inspected id",
           "circa": "item-disjoint and Y-answer-text-disjoint from DEV/TEST circa; SAME source and scenario types as the eval set (not family-disjoint)",
           "pub_task4 / pub_task5 / pub_task11 / pub_task12": "task-disjoint from the PUB eval tasks 2/3/6; contexts shown in the inspected pub6 sample removed; task 11 premise sentences that occur in eval IMPPRES presup removed",
           "impl1k": "different source from every DEV/TEST set; only rows never inspected",
           "imppres_presup": "item-disjoint and premise-sentence-disjoint from DEV/TEST/inspected; NOT family-disjoint: 99 of 100 paradigms are shared with the DEV/TEST sample",
           "folio": "train items only; the 200 dev items and all stories with a dev item removed; FOLIO validation (= TEST) has no story overlap with train",
           "gsm8k": "train split minus the 150 dev items (TEST is the official test split)",
           "unavailable": "LUDWIG train (release has dev/test only, all in DEV/TEST/inspected); NOPE (all 692 items in DEV/TEST); ImplicatureX rec/can (all inspected/used); the lead's list therefore cannot be met for these three"}}
blob = json.dumps(spl, sort_keys=True); spl["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
json.dump(spl, open(SPL / "sit_train_families.json", "w"), indent=1)
print(json.dumps({"pool_sizes": spl["pool_sizes"], "family_pool_sizes": spl["family_pool_sizes"], "availability": why, "sha256": spl["sha256"]}, indent=1))
