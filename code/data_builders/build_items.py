#!/usr/bin/env python3
"""Normalise the raw pragmatics stimulus sets in exp/data/raw/ into one JSONL schema (exp/data/items/*.jsonl).

Run:  python3 exp/data/build_items.py            (pandas + pyarrow only; deterministic; writes items/ and manifest.json)

Item schema (one JSON object per line)
  id          str   globally unique  "<source>.<subset>.<index>"
  phenomenon  str   scalar_implicature | underinformative | presupposition | indirect_speech | indirect_answer |
                    irony | metaphor | humour | deceit | maxims | coherence | sarcasm | ... (see PROVENANCE.md)
  source      str   imppres | bn_authored | hu2022 | ruis2022 | pub
  subset      str   trigger / task / phenomenon code inside the source
  role        str   target (a literal and a pragmatic reading can differ) | control | pending_validation (not yet signed off; never in reported numbers)
  context     str   story / dialogue / premise ("" if none)
  question    str   the question put to the model (the *default* framing)
  question_alt str|None  alternative framing (ablation of task demands), same options
  options     list[str]  canonical order; the harness permutes them per seed
  option_kinds list[str] role of each option (pragmatic | literal | entailment | neutral | ... | distractor)
  gold        int   index of the gold answer (= the pragmatic reading whenever one is defined)
  literal     int|None  index of the literal-reading answer when the source defines one
  meta        dict  source-specific fields (human accuracy, trigger type, spec_relation, ...)
  gold_set    list  optional: option indices that all count as the pragmatic response (ternary: 'a bit wrong' or 'completely wrong')
  labels/shuffle/scale  optional answer labels (default A,B,..), whether options may be permuted (False for ordinal scales), 1-7 scale flag
  split       str   "dev" | "test" (50:50, stratified, seed 0; pairs/scenarios/scales kept together). Reported numbers use test only.
  variants    dict  optional framings that replace context/question/options/option_kinds/gold/literal, keyed by name
              ("felicity": Yes/No judgement of what a speaker believes / whether the sentence is a good description;
               Yes = pragmatic reading for IMPPRES targets, No = pragmatic reading for underinformative sentences)
"""
import ast, csv, hashlib, io, json, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw", HERE / "items"
OUT.mkdir(exist_ok=True)
manifest = {}


def _split_key(it):
    m = it.get("meta", {})
    if "pair_index" in m: return f"pair{m['pair_index']}", "bn"
    if "scenario" in m: return f"scn{m['scenario']}", f"n{m.get('n', 0)}"
    if "ix_id" in m: return m["ix_id"], m["type"]
    if it["source"] == "nope": return f"nope{m['uid_base']}", "nope"
    if it["source"] == "imppres_xl": return m["orig_id"], "xl"
    if it["source"] == "figqa": return f"fq{m['qid']}", "figqa"
    if str(it["id"]).startswith("pub6."): return f"pub6.{m['item_no']}" if m["pub_task"] in (2, 3) else it["id"], f"t{m['pub_task']}"
    if it["source"] == "scalar_diversity": return f"{m['dataset']}|{m['scale']}", m["dataset"]
    if it["source"] == "hu2022": return it["id"], it["subset"]
    return it["id"], f"{it['subset']}|{it['role']}"


def assign_splits(items, seed=0):
    """dev/test 50:50, stratified, deterministic (sha256 of seed|group); items of one group (pair, scenario, scale) share a split."""
    strata = {}
    for it in items:
        g, st = _split_key(it)
        strata.setdefault(st, {}).setdefault(g, []).append(it)
    for st, groups in strata.items():
        order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}|{g}".encode()).hexdigest())
        for i, g in enumerate(order):
            for it in groups[g]:
                it["split"] = "dev" if i % 2 == 0 else "test"


def write(name, items):
    assign_splits(items)
    p = OUT / f"{name}.jsonl"
    with open(p, "w") as f:
        for it in items:
            assert set(it["options"]) and len(it["options"]) == len(it["option_kinds"]) and 0 <= it["gold"] < len(it["options"]), it["id"]
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    roles = Counter(i["role"] for i in items)
    manifest[name] = {"n": len(items), "roles": dict(roles), "sha256_16": h}
    print(f"{name:34s} n={len(items):5d} {dict(roles)}")


def item(**kw):
    kw.setdefault("context", ""); kw.setdefault("question_alt", None); kw.setdefault("literal", None); kw.setdefault("meta", {}); kw.setdefault("variants", {})
    return kw


# ----------------------------------------------------------------------------------------------------------------
# 1. IMPPRES (Jeretic et al. 2020): scalar implicature (7 triggers x 1200) and presupposition
# ----------------------------------------------------------------------------------------------------------------
NLI_OPTS = ["Yes, the second sentence must be true if the first is true.",
            "Not necessarily: the second sentence may be true or false.",
            "No, the second sentence must be false if the first is true."]
NLI_KINDS = ["entailment", "neutral", "contradiction"]
YN = ["Yes", "No"]


def felicity_nli(premise, hypothesis, gold, literal):
    """Yes/No 'would you conclude that the speaker believes ...' framing of a 3-way NLI item.
    gold 0 (entail): ask whether the speaker believes the second statement is true, Yes = gold;
    gold 2 (contradict): ask whether the speaker believes it is false, Yes = gold;
    gold 1 (neutral): ask 'true', No = gold.  For scalar targets Yes is always the pragmatic reading and No the literal one,
    so answers are confounded with yes-bias unless read against the control items (neutral / contradiction controls: gold No or Yes)."""
    what = "false" if gold == 2 else "true"
    ctx = f'A speaker says: "{premise}"\nSecond statement: "{hypothesis}"'
    q = f"Would you conclude that the speaker believes the second statement is {what}? Answer Yes or No."
    g = 1 if gold == 1 else 0
    kinds = ["pragmatic", "literal"] if literal is not None and literal != gold else ["yes", "no"]
    return {"context": ctx, "question": q, "options": YN, "option_kinds": kinds, "gold": g,
            "literal": (1 if literal is not None and literal != gold else (g if literal is not None else None))}


IMPPRES_SCALAR = {  # dir -> (file stem, label used in ids)
    "implicature_quantifiers": "quantifiers", "implicature_connectives": "connectives", "implicature_modals": "modals",
    "implicature_gradable_adjective": "gradable_adjective", "implicature_gradable_verb": "gradable_verb",
    "implicature_numerals_10_100": "numerals_10_100", "implicature_numerals_2_3": "numerals_2_3"}


def imppres_scalar():
    items = []
    for d, stem in IMPPRES_SCALAR.items():
        df = pd.read_parquet(RAW / "imppres" / d / f"{stem}-00000-of-00001.parquet")
        for i, r in df.iterrows():
            log, prag = int(r.gold_label_log), int(r.gold_label_prag)
            diag = log != prag
            # 'question' = formal NLI framing; 'question_alt' = speaker framing (task-demand ablation)
            ctx = f"First sentence: {r.premise}\nSecond sentence: {r.hypothesis}"
            q = "Assume the first sentence is true. Does the second sentence follow?"
            alt = ("Someone says the first sentence in an ordinary conversation. Would a cooperative listener "
                   "conclude that the second sentence is true, false, or that it is left open?")
            items.append(item(id=f"imppres.{stem}.{i}", phenomenon="scalar_implicature", source="imppres", subset=stem,
                              role="target" if diag else "control", context=ctx, question=q, question_alt=alt,
                              options=NLI_OPTS, option_kinds=NLI_KINDS, gold=prag, literal=log,
                              meta={"trigger": stem, "spec_relation": r.spec_relation, "lexemes": r.lexemes,
                                    "gold_label_log": log, "gold_label_prag": prag, "premise": r.premise, "hypothesis": r.hypothesis},
                              variants={"felicity": felicity_nli(r.premise, r.hypothesis, prag, log)}))
    write("imppres_scalar", items)
    # diagnostic-only view: items where the logical and the pragmatic label differ
    write("imppres_scalar_target", [x for x in items if x["role"] == "target"])


def imppres_presup():
    df = pd.read_parquet(RAW / "imppres" / "presupposition_all_n_presupposition" / "all_n_presupposition-00000-of-00001.parquet")
    items = []
    for i, r in df.iterrows():
        ctx = f"First sentence: {r.premise}\nSecond sentence: {r.hypothesis}"
        items.append(item(id=f"imppres.all_n_presupposition.{i}", phenomenon="presupposition", source="imppres", subset="all_n_presupposition",
                          role="target", context=ctx, question="Assume the first sentence is true. Does the second sentence follow?",
                          options=NLI_OPTS, option_kinds=NLI_KINDS, gold=int(r.gold_label),
                          meta={"trigger": r.trigger, "presupposition": r.presupposition, "paradigmID": int(r.paradigmID)},
                          variants={"felicity": felicity_nli(r.premise, r.hypothesis, int(r.gold_label), None)}))
    write("imppres_presupposition", items)


# ----------------------------------------------------------------------------------------------------------------
# 2. Bott & Noveck-style underinformative sentences (AUTHORED by us; not a published stimulus set)
#    "Some elephants are mammals": logically true, pragmatically false ("some" implicates "not all").
#    X, Y (true hypernym), P (predicate true of some but not all X), Z (a category no X belongs to)
# ----------------------------------------------------------------------------------------------------------------
BN = [
    ("elephants", "mammals", "live in Asia", "insects"), ("sparrows", "birds", "live in cities", "mammals"),
    ("roses", "flowers", "are red", "animals"), ("tulips", "flowers", "are yellow", "vegetables"),
    ("oaks", "trees", "grow in Europe", "machines"), ("salmon", "fish", "are farmed", "birds"),
    ("sharks", "fish", "are dangerous to humans", "mammals"), ("spiders", "arachnids", "build webs", "birds"),
    ("whales", "mammals", "live in cold water", "birds"), ("penguins", "birds", "live in Antarctica", "mammals"),
    ("apples", "fruits", "are green", "vegetables"), ("oranges", "fruits", "are seedless", "vegetables"),
    ("carrots", "vegetables", "are purple", "fruits"), ("potatoes", "vegetables", "are red", "fruits"),
    ("hammers", "tools", "have wooden handles", "animals"), ("wrenches", "tools", "are adjustable", "fruits"),
    ("pianos", "instruments", "are black", "vehicles"), ("guitars", "instruments", "are electric", "vegetables"),
    ("cars", "vehicles", "are red", "animals"), ("bicycles", "vehicles", "have baskets", "fruits"),
    ("trains", "vehicles", "run on electricity", "animals"), ("airplanes", "vehicles", "have propellers", "plants"),
    ("cats", "mammals", "are black", "plants"), ("dogs", "mammals", "are brown", "birds"),
    ("horses", "mammals", "are white", "reptiles"), ("cows", "mammals", "are brown", "birds"),
    ("frogs", "amphibians", "are green", "mammals"), ("snakes", "reptiles", "are venomous", "birds"),
    ("crocodiles", "reptiles", "live in Africa", "mammals"), ("lizards", "reptiles", "can change color", "mammals"),
    ("bees", "insects", "make honey", "mammals"), ("ants", "insects", "live underground", "birds"),
    ("butterflies", "insects", "are orange", "mammals"), ("rivers", "bodies of water", "flow into the sea", "animals"),
    ("lakes", "bodies of water", "are salty", "animals"), ("mountains", "landforms", "are volcanoes", "fruits"),
    ("islands", "landforms", "are inhabited", "insects"), ("pizzas", "foods", "have mushrooms", "metals"),
    ("pears", "fruits", "are yellow", "vegetables"), ("soups", "foods", "are served cold", "vehicles"),
    ("sandwiches", "foods", "contain cheese", "trees"), ("shirts", "items of clothing", "have pockets", "fish"),
    ("hats", "items of clothing", "have brims", "vegetables"), ("novels", "books", "are mysteries", "fruits"),
    ("owls", "birds", "are white", "mammals"),
]
TF = ["True", "False"]

# 35 further triples (support, 2026-09-30, criterion 3'; screened by two LLM raters: 31 OK, 3 fixes applied, 1 pending). Imported with
# role "pending_validation" until both raters signed off; SIGNED OFF 2026-09-30 (support + lead) -> normal roles, meta.cohort = "new35".
BN_NEW = [
    ("tigers", "mammals", "live in India", "birds"), ("rabbits", "mammals", "are white", "birds"),
    ("wolves", "mammals", "live in Canada", "birds"), ("eagles", "birds", "live in Scotland", "mammals"),
    ("parrots", "birds", "can talk", "mammals"), ("chickens", "birds", "lay brown eggs", "mammals"),
    ("trout", "fish", "live in lakes", "mammals"), ("tuna", "fish", "live in the Atlantic", "birds"),
    ("beetles", "insects", "are black", "mammals"), ("mosquitoes", "insects", "carry diseases", "birds"),
    ("wasps", "insects", "are yellow", "mammals"), ("turtles", "reptiles", "live in the sea", "mammals"),
    ("alligators", "reptiles", "live in Florida", "birds"), ("toads", "amphibians", "are brown", "mammals"),
    ("lemons", "fruits", "are grown in Spain", "vegetables"), ("bananas", "fruits", "are green", "animals"),
    ("grapes", "fruits", "are purple", "vegetables"), ("strawberries", "fruits", "are grown in greenhouses", "vegetables"),
    ("onions", "vegetables", "are red", "fruits"), ("cabbages", "vegetables", "are purple", "fruits"),
    ("maples", "trees", "turn red in autumn", "machines"), ("violins", "instruments", "are very expensive", "vehicles"),
    ("trumpets", "instruments", "are silver", "vegetables"), ("vans", "vehicles", "are yellow", "animals"),
    ("trucks", "vehicles", "are red", "fruits"), ("motorcycles", "vehicles", "have sidecars", "animals"),
    ("sweaters", "items of clothing", "are made of wool", "fish"), ("socks", "items of clothing", "are striped", "fruits"),
    ("sofas", "pieces of furniture", "are made of leather", "animals"), ("bookcases", "pieces of furniture", "are made of oak", "fish"),
    ("refrigerators", "appliances", "have ice makers", "animals"), ("toasters", "appliances", "have four slots", "fruits"),
    ("muffins", "foods", "contain blueberries", "vehicles"),  # PENDING the writer's confirmation; drop if rejected
    ("newspapers", "publications", "are published daily", "animals"), ("burgers", "foods", "are vegetarian", "metals"),
]

# sentence types of the plan (Bott & Noveck 2004; §3 A1): id-stable subsets keep their old names
#   T1 underinformative "Some X are Y" (logically true, pragmatically odd)    -> subset "underinformative"   [target]
#   T2 "Some Y are X" (true, informative)                                      -> "t2_some_rev"
#   T3 "Some X are Z" (false)                                                  -> "ctrl_false"
#   T4 "All X are Y" (true)                                                    -> "ctrl_all_true"
#   T5 "All Y are X" (false)                                                   -> "t5_all_rev"
#   T6 "All X are Z" (false)                                                   -> "t6_all_false"
#   T0 "Some X P" (true, informative partial predicate; extra control)         -> "ctrl_some_true"
ST = {"underinformative": "T1", "t2_some_rev": "T2", "ctrl_false": "T3", "ctrl_all_true": "T4", "t5_all_rev": "T5", "t6_all_false": "T6", "ctrl_some_true": "T0"}
KATSOS = ["Completely right", "A bit wrong", "Completely wrong"]           # Katsos & Bishop 2011 three-way judgement
GRADED7 = [str(i) for i in range(1, 8)]


def _bn_variants(cond, sentence, logical_true, prag_true):
    """felicity (Yes/No good description), ternary (Katsos & Bishop 3-way) and graded (1-7 goodness) framings of one sentence."""
    ctx = f"Sentence: {sentence}"
    good = logical_true and prag_true          # a good description of the facts under the pragmatic reading
    fel = {"context": ctx, "question": "Is this sentence a good way to describe the facts? Answer Yes or No.", "options": YN,
           "option_kinds": ["literal", "pragmatic"] if cond == "underinformative" else ["yes", "no"],
           "gold": 0 if good else 1, "literal": 0 if logical_true else 1}
    # ternary: pragmatic = 'a bit wrong' OR 'completely wrong' for T1; true-and-good = 'completely right'; false = 'completely wrong'
    if cond == "underinformative":
        ter_gold, ter_set, ter_kinds = 1, [1, 2], ["literal", "pragmatic", "pragmatic_strong"]
    elif good:
        ter_gold, ter_set, ter_kinds = 0, [0], ["right", "a_bit_wrong", "wrong"]
    else:
        ter_gold, ter_set, ter_kinds = 2, [2], ["right", "a_bit_wrong", "wrong"]
    ter = {"context": ctx, "question": "Is what the speaker said completely right, a bit wrong, or completely wrong?", "options": KATSOS,
           "option_kinds": ter_kinds, "gold": ter_gold, "gold_set": ter_set, "literal": 0 if logical_true else 2}
    gra = {"context": ctx, "question": "On a scale from 1 (very poor) to 7 (very good), how good a description of the facts is this sentence? Answer with a number.",
           "options": GRADED7, "option_kinds": ["scale"] * 7, "gold": 0 if not good else 6, "literal": None, "scale": True, "labels": "1234567",
           "shuffle": False}
    return {"felicity": fel, "ternary": ter, "graded": gra}


def bn_items():
    items = []
    for k, (x, y, p, z) in enumerate(BN + BN_NEW):
        pending = False   # BN_NEW signed off by both LLM raters (support/lead, 2026-09-30); provenance kept in meta.cohort
        conds = [  # (cond, sentence, logical truth, pragmatic truth, role)
            ("underinformative", f"Some {x} are {y}.", True, False, "target"),
            ("t2_some_rev", f"Some {y} are {x}.", True, True, "control"),
            ("ctrl_false", f"Some {x} are {z}.", False, False, "control"),
            ("ctrl_all_true", f"All {x} are {y}.", True, True, "control"),
            ("t5_all_rev", f"All {y} are {x}.", False, False, "control"),
            ("t6_all_false", f"All {x} are {z}.", False, False, "control"),
            ("ctrl_some_true", f"Some {x} {p}.", True, True, "control")]
        for cond, s, lt, pt, role in conds:
            v = _bn_variants(cond, s, lt, pt)
            items.append(item(id=f"bn.{cond}.{k}", phenomenon="underinformative", source="bn_authored", subset=cond,
                              role="pending_validation" if pending else role,
                              context=f"Sentence: {s}", question="Is the sentence true or false?",
                              question_alt="A speaker says this sentence to a listener. Is what the speaker said true or false?",
                              options=TF, option_kinds=["literal" if lt else "pragmatic", "pragmatic" if not pt else "literal"] if cond == "underinformative" else ["true", "false"],
                              gold=0 if pt else 1, literal=0 if lt else 1,
                              meta={"category": x, "logical_truth": lt, "pragmatic_truth": pt, "sentence": s, "sentence_type": ST[cond],
                                    "pair_index": k, "cohort": "new35" if k >= len(BN) else "orig45", "intended_role": role, "y": y, "z": z, "partial": p},
                              variants=v))
    write("bn_underinformative", items)
    write("bn_underinformative_target", [i for i in items if i["role"] == "target"])


# ----------------------------------------------------------------------------------------------------------------
# 2b. A1-ctx (support, notes/A1_CTX_ITEMS.md §3-§4; Goodman & Stuhlmueller 2013-style speaker-knowledge manipulation)
#     50 scenarios x {full, partial} access x {felicity (speaker frame), inference (listener frame)}; KSI = R_full - R_partial.
#     role: full access = target (pragmatic answer differs from literal), partial access = control (Yes is both literal and pragmatic).
#     Status: FINAL (both LLM raters, 2026-09-30); a no-utterance prior probe (a1ctx_prior) runs first and excludes scenarios answered "not all".
# ----------------------------------------------------------------------------------------------------------------
A1CTX = [
 ("Mia", 5, "apples", "apple", "in closed boxes on a shelf", "are red", "was red", "were red"),
 ("Tom", 4, "letters", "letter", "in sealed envelopes on a desk", "are signed", "was signed", "were signed"),
 ("Ana", 6, "cookies", "cookie", "in closed tins in a cupboard", "are broken", "was broken", "were broken"),
 ("Ben", 5, "rabbits", "rabbit", "in covered hutches in a yard", "are asleep", "was asleep", "were asleep"),
 ("Lena", 8, "helmets", "helmet", "in locked lockers in a changing room", "are cracked", "was cracked", "were cracked"),
 ("Omar", 4, "batteries", "battery", "in the drawers of a workbench", "are dead", "was dead", "were dead"),
 ("Sofia", 6, "fish", "fish", "in covered tanks in a pet shop", "are striped", "was striped", "were striped"),
 ("Raj", 5, "bottles", "bottle", "in a crate in a dark cellar", "are empty", "was empty", "were empty"),
 ("Kim", 3, "printers", "printer", "on different floors of an office", "are out of paper", "was out of paper", "were out of paper"),
 ("Leo", 6, "tickets", "ticket", "in sealed envelopes in a drawer", "are for Friday", "was for Friday", "were for Friday"),
 ("Nora", 5, "plants", "plant", "in a greenhouse with frosted glass", "are flowering", "was flowering", "were flowering"),
 ("Sam", 4, "cars", "car", "in a car park", "are unlocked", "was unlocked", "were unlocked"),
 ("Eva", 8, "eggs", "egg", "in a closed carton in a fridge", "are cracked", "was cracked", "were cracked"),
 ("Hugo", 5, "laptops", "laptop", "in the bags of a school trolley", "are charged", "was charged", "were charged"),
 ("Ivy", 6, "windows", "window", "on the upper floor of a house", "are open", "was open", "were open"),
 ("Jack", 4, "tires", "tire", "on a parked van", "are dirty", "was dirty", "were dirty"),
 ("Maya", 5, "cupcakes", "cupcake", "in closed paper boxes at a bakery", "are frosted", "was frosted", "were frosted"),
 ("Noah", 6, "parcels", "parcel", "in the back of a delivery van", "are damaged", "was damaged", "were damaged"),
 ("Olga", 5, "emails", "email", "in an unread folder", "are from the manager", "was from the manager", "were from the manager"),
 ("Paul", 4, "bikes", "bike", "in a shed behind a school", "have flat tires", "had a flat tire", "had flat tires"),
 ("Rosa", 6, "lamps", "lamp", "in different rooms of a house", "are switched on", "was switched on", "were switched on"),
 ("Tariq", 5, "doors", "door", "along a long hotel corridor", "are locked", "was locked", "were locked"),
 ("Uma", 3, "cats", "cat", "in carriers at a vet clinic", "have collars", "had a collar", "had collars"),
 ("Victor", 6, "puzzles", "puzzle", "in closed boxes in a toy shop", "are missing pieces", "was missing pieces", "were missing pieces"),
 ("Wen", 5, "bags", "bag", "in a storage room", "are torn", "was torn", "were torn"),
 ("Xavier", 4, "pens", "pen", "in a pencil case", "are out of ink", "was out of ink", "were out of ink"),
 ("Yara", 6, "phones", "phone", "in a box at a repair shop", "have cracked screens", "had a cracked screen", "had cracked screens"),
 ("Zoe", 5, "books", "book", "on a list of library loans", "are overdue", "was overdue", "were overdue"),
 ("Adam", 8, "seats", "seat", "in the front row of a theatre", "are taken", "was taken", "were taken"),
 ("Bella", 4, "pumpkins", "pumpkin", "in a dark barn", "are rotten", "was rotten", "were rotten"),
 ("Carlos", 6, "boats", "boat", "moored along a dock", "are leaking", "was leaking", "were leaking"),
 ("Dina", 5, "class photos", "class photo", "in an album on a school website", "are blurry", "was blurry", "were blurry"),
 ("Elif", 3, "ovens", "oven", "in a restaurant kitchen", "are hot", "was hot", "were hot"),
 ("Farid", 6, "oranges", "orange", "in closed paper bags", "are moldy", "was moldy", "were moldy"),
 ("Grace", 5, "statues", "statue", "under dust sheets in a museum", "are painted", "was painted", "were painted"),
 ("Hana", 4, "suitcases", "suitcase", "on a baggage cart", "are heavy", "was heavy", "were heavy"),
 ("Ian", 6, "trees", "tree", "along a forest path", "have nests in them", "had a nest in it", "had nests in them"),
 ("Julia", 5, "shirts", "shirt", "in a closed wardrobe", "are ironed", "was ironed", "were ironed"),
 ("Kofi", 4, "wells", "well", "in a village", "are dry", "was dry", "were dry"),
 ("Lara", 6, "candles", "candle", "in lanterns in a garden", "are melted", "was melted", "were melted"),
 ("Marco", 5, "radios", "radio", "in boxes at a flea market", "are working", "was working", "were working"),
 ("Nadia", 8, "cups", "cup", "in a closed cabinet", "are chipped", "was chipped", "were chipped"),
 ("Oscar", 4, "paintings", "painting", "wrapped in paper in a gallery store", "are framed", "was framed", "were framed"),
 ("Priya", 6, "tomatoes", "tomato", "in a covered basket", "are ripe", "was ripe", "were ripe"),
 ("Quinn", 5, "clocks", "clock", "in different rooms of a school", "are fast", "was fast", "were fast"),
 ("Ruth", 3, "dogs", "dog", "in kennels at a shelter", "are muddy", "was muddy", "were muddy"),
 ("Sven", 6, "light bulbs", "light bulb", "in boxes in a warehouse", "are burnt out", "was burnt out", "were burnt out"),
 ("Tara", 5, "files", "file", "in a shared folder on a computer", "are corrupted", "was corrupted", "were corrupted"),
 ("Umar", 4, "bananas", "banana", "in closed paper bags", "are bruised", "was bruised", "were bruised"),
 ("Vera", 6, "plates", "plate", "stacked in a closed dishwasher", "are dirty", "was dirty", "were dirty"),
]
NUMW = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight"}
YESNO = ["Yes", "No"]


def a1ctx_items():
    fel_items, inf_items, prior_items = [], [], []
    for k, (name, n, pl, sg, where, prop, past_sg, past_pl) in enumerate(A1CTX):
        N = NUMW[n]
        setting = f"There are {N} {pl} {where}."
        utt = f"Some of the {pl} {prop}."
        access = {"full": f"{name} checks all {N} of the {pl}.",
                  "partial": f"{name} has time to check only two of the {pl}; the other {NUMW[n - 2]} {'is' if n - 2 == 1 else 'are'} not checked."}
        obs = {"full": f"Every {sg} that {name} checked {past_sg}.", "partial": f"Both {pl} that {name} checked {past_pl}."}
        fel_q = f'Given what {name} found out, is "{utt[:-1]}" a good way for {name} to describe the {N} {pl}? Answer Yes or No.'
        inf_q = f"Based on what {name} said, is it possible that all {N} {pl} {prop}? Answer Yes or No."
        for acc in ("full", "partial"):
            full = acc == "full"
            base = dict(phenomenon="speaker_knowledge", source="a1ctx_authored", role="target" if full else "control",
                        meta={"scenario": k, "access": acc, "name": name, "n": n, "noun": pl, "property": prop, "utterance": utt, "status": "final"})
            kinds = ["literal", "pragmatic"] if full else ["yes", "no"]
            g = 1 if full else 0
            # ---- speaker frame (felicity)
            ctx_f = f'{setting} {access[acc]} {obs[acc]} {name} says: "{utt}"'
            var_f = {
                "ternary": {"context": ctx_f, "question": f'Given what {name} found out, how would you judge the statement "{utt[:-1]}" as a description of the {N} {pl}?',
                            "options": ["True and a good description", "True but misleading", "False"],
                            "option_kinds": ["literal", "pragmatic", "comprehension_failure"], "gold": 1 if full else 0, "literal": 0},
                "graded": {"context": ctx_f, "question": f'Given what {name} found out, how good is "{utt[:-1]}" as a description of the {N} {pl}? 1 = very bad, 7 = very good. Answer with a number.',
                           "options": GRADED7, "option_kinds": ["scale"] * 7, "gold": 0 if full else 6, "literal": None, "scale": True, "labels": "1234567", "shuffle": False}}
            fel_items.append(item(id=f"a1ctx.{k}.felicity.{acc}", subset=f"felicity_{acc}", context=ctx_f, question=fel_q, options=YESNO,
                                  option_kinds=kinds, gold=g, literal=0, variants=var_f, **base))
            # ---- listener frame (inference; access only)
            ctx_i = f'{setting} {access[acc]} {name} says: "{utt}"'
            var_i = {
                "ternary": {"context": ctx_i, "question": f"Based on what {name} said, what do you conclude about the {N} {pl}?",
                            "options": [f"Not all of them {prop}", f"Possibly all of them {prop}", f"None of them {prop}"],
                            "option_kinds": ["pragmatic", "literal", "comprehension_failure"], "gold": 0 if full else 1, "literal": 1},
                "graded": {"context": ctx_i, "question": f"Based on what {name} said, how likely is it that all {N} {pl} {prop}? 1 = very unlikely, 7 = very likely. Answer with a number.",
                           "options": GRADED7, "option_kinds": ["scale"] * 7, "gold": 0 if full else 4, "literal": None, "scale": True, "labels": "1234567", "shuffle": False}}
            inf_items.append(item(id=f"a1ctx.{k}.inference.{acc}", subset=f"inference_{acc}", context=ctx_i, question=inf_q, options=YESNO,
                                  option_kinds=kinds, gold=g, literal=0, variants=var_i, **base))
        # extra control (support §2): full access + "All of the X are P" -> felicity gold Yes (checks that quantified descriptions are not simply rejected)
        ctx_a = f'{setting} {access["full"]} {obs["full"]} {name} says: "All of the {pl} {prop}."'
        fel_items.append(item(id=f"a1ctx.{k}.felicity.ctrl_all", phenomenon="speaker_knowledge", source="a1ctx_authored", subset="felicity_ctrl_all",
                              role="control", context=ctx_a, question=f'Given what {name} found out, is "All of the {pl} {prop}" a good way for {name} to describe the {N} {pl}? Answer Yes or No.',
                              options=YESNO, option_kinds=["yes", "no"], gold=0, literal=0,
                              meta={"scenario": k, "access": "full", "utterance": f"All of the {pl} {prop}.", "status": "final"}))
        # no-utterance prior probe: is 'all' plausible from the setting alone? (a scenario answered 'No' = prior against 'all' -> excluded)
        prior_items.append(item(id=f"a1ctx.{k}.prior", phenomenon="speaker_knowledge", source="a1ctx_authored", subset="prior_probe", role="control",
                                context=f"{setting} {access['full']}", question=f"Is it possible that all {N} {pl} {prop}? Answer Yes or No.",
                                options=YESNO, option_kinds=["yes", "no"], gold=0, literal=0,
                                meta={"scenario": k, "status": "final", "note": "answer No without the utterance => scenario excluded from A1-ctx"}))
    write("a1ctx_felicity", fel_items)
    write("a1ctx_inference", inf_items)
    write("a1ctx_prior", prior_items)


# ----------------------------------------------------------------------------------------------------------------
# 3. Hu et al. 2022 / Floyd, Jouravlev et al. (in prep): 20-25 items per phenomenon, 4 typed options, human data
# ----------------------------------------------------------------------------------------------------------------
HU = {  # phenomenon file stem -> (our name, item code prefix, human csv, question fallback)
    "IndirectSpeech": ("indirect_speech", "Human_IndirectSpeech.csv"), "Irony": ("irony", "Human_Irony.csv"),
    "Metaphor": ("metaphor", "Human_Metaphor.csv"), "Humour": ("humour", "Human_Humour.csv"),
    "Deceits": ("deceit", "Human_Deceits.csv"), "Maxims": ("maxims", "Human_Maxims.csv"),
}  # CoherenceInference is a binary coherence judgement without typed options: not used


def read_tsv(path):
    """Tab-separated, mixed CR/CRLF/LF line endings, quotes doubled; empty rows dropped."""
    t = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return [r for r in csv.reader(io.StringIO(t), delimiter="\t") if r]


COMPLEX = re.compile(r"^(Correct|Incorrect)[A-Za-z]+$")


def _num(r):
    """(item, option) integer columns: files either start with a phenomenon code or directly with the item number."""
    o = 0 if r[0].strip().isdigit() else 1
    return int(r[o]), int(r[o + 1]) if len(r) > o + 1 and r[o + 1].strip().isdigit() else None


def hu_items():
    base = RAW / "hu2022" / "fj_raw_data"
    n_all = {}
    for stem, (name, hcsv) in HU.items():
        scen = {}
        for r in read_tsv(base / f"{stem}Scenarios.txt"):
            scen[int(r[-2])] = r[-1].strip()
        opts = defaultdict(dict)
        for r in read_tsv(base / f"{stem}Options.txt"):
            it, k = _num(r)
            opts[it][k] = r[-1].strip()
        keys = {}
        for r in read_tsv(base / f"{stem}KeyComplex.txt"):
            it, k = _num(r)
            lab = [c.strip() for c in r[2:] if COMPLEX.match(c.strip())]
            keys[(it, k)] = lab[0] if lab else ""
        # human data
        hd = pd.read_csv(RAW / "hu2022" / "human_data" / "fj" / hcsv)
        hn = hd.groupby("itemNum").Correct.agg(["mean", "count"])
        hchoice = hd.groupby("itemNum").OptionChosen.value_counts(normalize=True).unstack(fill_value=0)
        items = []
        for it, text in sorted(scen.items()):
            o = opts.get(it, {})
            if not o:
                continue
            nums = sorted(o)
            kinds = [keys.get((it, k), "") or "unknown" for k in nums]
            gold = [i for i, kd in enumerate(kinds) if kd.startswith("Correct")]
            if len(gold) != 1:
                continue
            lit = [i for i, kd in enumerate(kinds) if kd == "IncorrectLiteral"]
            # split "story ... question" so the question is the final sentence
            m = re.search(r"([^.?!\"]*\?)\s*$", text)
            q = m.group(1).strip() if m else "Which option is best?"
            ctx = text[: m.start()].strip() if m else text
            hm = {"human_acc": float(hn.loc[it, "mean"]) if it in hn.index else None,
                  "human_n": int(hn.loc[it, "count"]) if it in hn.index else 0,
                  "human_choice": {c: float(hchoice.loc[it, c]) for c in hchoice.columns} if it in hchoice.index else {},
                  "item_no": it, "phenomenon_hu": stem}
            items.append(item(id=f"hu2022.{stem}.{it}", phenomenon=name, source="hu2022", subset=stem, role="target",
                              context=ctx, question=q, options=[o[k] for k in nums], option_kinds=kinds, gold=gold[0],
                              literal=lit[0] if lit else None, meta=hm))
        n_all[stem] = len(items)
        write(f"hu2022_{name}", items)
    return n_all


# ----------------------------------------------------------------------------------------------------------------
# 4. Ruis et al. 2022 "Goldilocks of pragmatic understanding": conversational implicature, yes/no
# ----------------------------------------------------------------------------------------------------------------
def ruis_items():
    tl = pd.read_csv(RAW / "ruis2022" / "type_labels.csv")
    type_by_key = {(re.sub(r"\s+", " ", str(u)).strip(), re.sub(r"\s+", " ", str(r)).strip()): (t, fk)
                   for u, r, t, fk in zip(tl.Utterance, tl.Response, tl.Type, tl["Factual knowledge"])}
    for split in ("test", "dev"):
        df = pd.read_csv(RAW / "ruis2022" / f"{split}_conversational_implicatures.csv")
        items, bad = [], 0
        for i, r in df.iterrows():
            u, resp = re.sub(r"\s+", " ", str(r["Context utterance"])).strip(), re.sub(r"\s+", " ", str(r["Response utterance"])).strip()
            imp = str(r["Implicature"]).strip().lower()
            if imp.startswith("yes"): gold = 0
            elif imp.startswith("no"): gold = 1
            else: bad += 1; continue
            t, fk = type_by_key.get((u, resp), (None, None))
            items.append(item(id=f"ruis2022.{split}.{i}", phenomenon="indirect_answer", source="ruis2022", subset=split, role="target",
                              context=f"Question: {u}\nResponse: {resp}", question="Does the response mean yes or no?",
                              options=["Yes", "No"], option_kinds=["pragmatic_yes", "pragmatic_no"], gold=gold,
                              meta={"implicature_raw": str(r["Implicature"]).strip(), "type": t, "factual_knowledge": fk}))
        write(f"ruis2022_{split}", items)
        if bad: print(f"  (ruis2022 {split}: skipped {bad} rows without yes/no implicature)")


# ----------------------------------------------------------------------------------------------------------------
# 5. PUB (Sravanthi et al. 2024, cfilt/PUB, MIT): 14 tasks; gold = "correct answer" string
# ----------------------------------------------------------------------------------------------------------------
PUB_PH = {1: "indirect_answer", 2: "indirect_answer", 3: "indirect_answer", 4: "implicature_recovery", 5: "sarcasm", 6: "sarcasm",
          7: "figurative", 8: "figurative", 9: "figurative", 10: "scalar_implicature", 11: "presupposition", 12: "presupposition",
          13: "deixis", 14: "metonymy"}


def pub_items():
    for t in range(1, 15):
        rows = [json.loads(l) for l in open(RAW / "pub" / f"t{t}" / f"task_{t}.jsonl")]
        items = []
        for r in rows:
            opts = [str(o).strip() for o in r["options"]]
            ca = str(r["correct answer"]).strip()
            if ca not in opts:
                continue
            items.append(item(id=f"pub.task{t}.{r['id']}", phenomenon=PUB_PH[t], source="pub", subset=f"task{t}", role="target",
                              context=r["pretext"].strip(), question="Choose the best option.", options=opts,
                              option_kinds=["option"] * len(opts), gold=opts.index(ca), meta={"pub_task": t}))
        write(f"pub_task{t:02d}", items)


# ----------------------------------------------------------------------------------------------------------------
# 6. A2 scalar diversity (van Tiel et al. 2016 vt16; Gotzner et al. 2018 g18; Pankratz & van Tiel 2021 pvt21; Ronai & Xiang 2022 rx22)
#    Materials as packaged by Hu et al. (github.com/jennhu/expectations-over-alternatives @ 50a7064290a841b81b2608524000b81a33ddc4b0,
#    cross-scale/human_data; no LICENSE file, cite the four papers + repo). One item = one sentence frame of one scale; the model is asked
#    whether it would conclude the speaker believes the full "weak, but not strong" construction (Yes = the scalar inference is drawn).
#    gold = Yes (index 0) by convention; the target is the *rate* R against meta.human_si (0-1 per scale), not accuracy.
# ----------------------------------------------------------------------------------------------------------------
def _split_construction(c):
    c = str(c).strip()
    i = c.lower().find(", but not")
    return (c[:i].strip().rstrip(",") + ".", c) if i > 0 else (None, c)


def a2_items():
    base = RAW / "scalar_diversity"
    def mk(ds, sid, sentence, construction, si, extra):
        weak, cons = _split_construction(construction)
        if weak is None or not sentence:
            return None
        weak = sentence.strip() if sentence else weak
        cons = cons[0].upper() + cons[1:]
        weak = weak[0].upper() + weak[1:]
        return item(id=f"a2.{ds}.{len(items_a2)}", phenomenon="scale_implicature", source="scalar_diversity", subset=ds, role="target",
                    context=f'A speaker says: "{weak}"',
                    question=f'Would you conclude that the speaker believes: "{cons}"? Answer Yes or No.',
                    question_alt=None, options=YN, option_kinds=["pragmatic", "literal"], gold=0, literal=1,
                    meta={"dataset": ds, "scale": sid, "human_si": si, "sentence": weak, "construction": cons, **extra})
    items_a2 = []
    vt = pd.read_csv(base / "vt16.csv")
    for _, r in vt.iterrows():
        for col, si in (("neutral_scalar_construction", r.si_neutral), ("nonneutral1_scalar_construction", r.si_nonneutral),
                        ("nonneutral2_scalar_construction", r.si_nonneutral), ("nonneutral3_scalar_construction", r.si_nonneutral)):
            weak, cons = _split_construction(r[col])
            if weak is None: continue
            cons = cons[0].upper() + cons[1:]
            items_a2.append(item(id=f"a2.vt16.{len(items_a2)}", phenomenon="scale_implicature", source="scalar_diversity", subset="vt16", role="target",
                                 context=f'A speaker says: "{weak[0].upper() + weak[1:]}"',
                                 question=f'Would you conclude that the speaker believes: "{cons}"? Answer Yes or No.',
                                 options=YN, option_kinds=["pragmatic", "literal"], gold=0, literal=1,
                                 meta={"dataset": "vt16", "scale": r.scale_id, "human_si": float(si) / 100.0, "template": col.split("_")[0],
                                       "weak": r.weak_scalemate, "strong": r.strong_scalemate, "category": r.category, "pos": r.get("adj")}))
    g = pd.read_csv(base / "g18.csv")
    for _, r in g.iterrows():
        weak, cons = _split_construction(r.scalar_construction)
        if weak is None: continue
        items_a2.append(item(id=f"a2.g18.{len(items_a2)}", phenomenon="scale_implicature", source="scalar_diversity", subset="g18", role="target",
                             context=f'A speaker says: "{weak[0].upper() + weak[1:]}"',
                             question=f'Would you conclude that the speaker believes: "{cons[0].upper() + cons[1:]}"? Answer Yes or No.',
                             options=YN, option_kinds=["pragmatic", "literal"], gold=0, literal=1,
                             meta={"dataset": "g18", "scale": r.scale_id, "human_si": float(r.si_rate), "weak": r.weak_adj, "strong": r.strong_adj, "pos": r.pos}))
    pv = pd.read_csv(base / "pvt21.csv")
    for _, r in pv.iterrows():
        weak, cons = _split_construction(r.combo_sentence)
        if weak is None: continue
        items_a2.append(item(id=f"a2.pvt21.{len(items_a2)}", phenomenon="scale_implicature", source="scalar_diversity", subset="pvt21", role="target",
                             context=f'A speaker says: "{str(r.weak_sentence).strip()}"',
                             question=f'Would you conclude that the speaker believes: "{cons[0].upper() + cons[1:]}"? Answer Yes or No.',
                             options=YN, option_kinds=["pragmatic", "literal"], gold=0, literal=1,
                             meta={"dataset": "pvt21", "scale": f"{r.weak_adj}/{r.strong_adj}", "human_si": float(r.SI_rate), "weak": r.weak_adj, "strong": r.strong_adj}))
    rx = pd.read_csv(base / "rx22.csv"); rt = pd.read_csv(base / "rx22_processed_exp1.csv")
    rate = {int(r.Item): float(r["SI percent (Exp 1)"]) / 100.0 for _, r in rx.iterrows()}
    seen = set()
    for _, r in rt.drop_duplicates(["Item", "Sentence frame"]).iterrows():
        key = (int(r.Item), r["Sentence frame"])
        weak, cons = _split_construction(r.scalar_construction)
        if key in seen or weak is None or int(r.Item) not in rate: continue
        seen.add(key)
        items_a2.append(item(id=f"a2.rx22.{len(items_a2)}", phenomenon="scale_implicature", source="scalar_diversity", subset="rx22", role="target",
                             context=f'A speaker says: "{str(r["Sentence frame"]).strip()}"',
                             question=f'Would you conclude that the speaker believes: "{cons[0].upper() + cons[1:]}"? Answer Yes or No.',
                             options=YN, option_kinds=["pragmatic", "literal"], gold=0, literal=1,
                             meta={"dataset": "rx22", "scale": f"{r.Weaker}/{r.Stronger}", "human_si": rate[int(r.Item)], "weak": r.Weaker, "strong": r.Stronger,
                                   "item_no": int(r.Item)}))
    write("a2_scalar_diversity", items_a2)


# ----------------------------------------------------------------------------------------------------------------
# 7. A6 ImplicatureX (Spinoso et al. 2026, arXiv:2607.25094; HF McGill-NLP/ImplicatureX @ 959e5e279611c877224d7573fe66fa9748f59aa4, MIT)
#    271 rows: synthetic_conversational 144, naturally_occurring_conversational 50, scalar 46, discourse 31. Per row three truth-judgement items about the
#    implicature statement: rec (utterance without cancellation -> True), can (utterance + cancellation -> False), prior (statement alone; the rate of
#    'True' without the utterance is the prior-only probe). The utterance field carries placeholders {implicature} / {cancellation}.
# ----------------------------------------------------------------------------------------------------------------
def ix_items():
    d = pd.read_csv(RAW / "implicaturex" / "implicatureX.csv", skiprows=1)
    TJ = ["True", "False"]; Q = "Based on the text above, is the following statement true or false? Answer True or False."
    out = []
    for n, r in d.iterrows():
        typ = r.implicature_type; ctx = "" if pd.isna(r.context) else str(r.context).strip()
        utt = str(r.utterance); canc = "" if pd.isna(r.cancellation) else str(r.cancellation)
        has_ph = "{cancellation}" in utt
        def fill(with_cancel):
            u = utt.replace("{implicature}", "").replace("{cancellation}", canc if with_cancel else "").replace("{context}", ctx)   # discourse rows embed the context sentence in the utterance
            if with_cancel and not has_ph: u = (u.rstrip() + " " + canc.strip()).strip()   # synthetic/scalar rows: the cancellation follows the utterance
            return u.strip()
        def text(u):
            if "{context}" in utt: return u            # context already placed inside the text
            if "{utterance}" in ctx: return ctx.replace("{utterance}", u)   # narrative contexts (jhu_*) embed the reply
            if typ == "synthetic_conversational": return (f"A: {ctx}\nB: {u}" if ctx else f"B: {u}")
            return (ctx + "\n" + u).strip()
        for kind, body, gold in (("rec", text(fill(False)), 0), ("can", text(fill(True)), 1), ("prior", "", 0)):
            out.append(item(id=f"ix.{r.id}.{kind}", phenomenon="implicature_cancellation", source="implicaturex", subset=typ,
                            role="control" if kind == "prior" else "target", context=body,
                            question=f"{Q}\nStatement: {r.implicature}", options=TJ, option_kinds=["implicature_holds", "implicature_does_not_hold"], gold=gold,
                            meta={"ix_id": str(r.id), "kind": kind, "implicature": str(r.implicature), "type": typ}))
    write("implicaturex", out)


# ----------------------------------------------------------------------------------------------------------------
# 8. B3 Fig-QA (Liu et al. 2022, github.com/nightingal3/Fig-QA @ fb638f847fb6ef0f336c6f6fd1c3d1b5b2c54b2d (UNVERIFIED sha: from the clone), MIT) and B4 PUB subset
#    B3: 500 items sampled (seed 0) from dev.csv (test labels are hidden); metaphor -> choose the correct paraphrase of two; minimal pairs share a qid (kept in one split).
#    B4: PUB tasks 2, 3, 4, 6, 11, 14 x 150 items (seed 0); tasks 2 and 3 are the SAME items without / with the human implied-meaning hint (aligned on item id).
# ----------------------------------------------------------------------------------------------------------------
def figqa_items():
    d = pd.read_csv(RAW / "figqa" / "dev.csv"); d = d[d["valid"] == 1].reset_index(drop=True)
    idx = list(range(len(d))); random.Random(0).shuffle(idx); out = []
    for n, i in enumerate(sorted(idx[:500])):
        r = d.iloc[i]
        out.append(item(id=f"figqa.dev.{i}", phenomenon="metaphor", source="figqa", subset="dev", role="target",
                        context=f"Sentence: {r.startphrase}.", question="Which of the two options best describes what the sentence means?",
                        options=[str(r.ending1), str(r.ending2)], option_kinds=["paraphrase", "paraphrase"], gold=int(r.labels), meta={"qid": str(r.qid), "startphrase": r.startphrase}))
    write("figqa", out)


def pub6_items():
    tasks = (2, 3, 4, 6, 11, 14); rows = {t: {r["id"].split(".")[-1]: r for r in map(json.loads, open(OUT / f"pub_task{t:02d}.jsonl"))} for t in tasks}   # keyed by the numeric item id
    ids23 = sorted(set(rows[2]) & set(rows[3]), key=int); rng = random.Random(0); rng.shuffle(ids23); pick23 = ids23[:150]
    out = []
    for t in tasks:
        if t in (2, 3): ids = pick23
        else:
            ids = sorted(rows[t], key=int); random.Random(t).shuffle(ids); ids = ids[:150]
        for i in ids:
            r = dict(rows[t][i]); r["id"] = f"pub6.task{t:02d}.{i}"; r["role"] = "target"
            r["meta"] = {**r.get("meta", {}), "pub_task": t, "item_no": i}; r.setdefault("variants", {})
            out.append(r)
    write("pub6", out)


# ----------------------------------------------------------------------------------------------------------------
# 12. Expansion sets (2026-10-04): Circa (indirect answers, Louis et al. 2020; HF google-research-datasets/circa) and a 1,000-item conversational-implicature set (HF tasksource/implicatures)
# ----------------------------------------------------------------------------------------------------------------
DIRECT = {"yes", "yeah", "yep", "yup", "no", "nope", "nah", "sure", "absolutely", "definitely", "certainly", "indeed"}
def circa_items(n_yes=200, n_no=200, n_mid=100):
    d = pd.read_parquet(RAW / "circa" / "train.parquet")
    d = d[d["goldstandard2"].isin([0, 1, 2])].copy()
    toks = d["answer-Y"].str.lower().str.findall(r"[a-z']+")
    d = d[~toks.apply(lambda t: any(w in DIRECT for w in t))].reset_index(drop=True)      # indirect answers only: no yes/no word anywhere in Y's reply
    out = []
    for g, k, nm in ((0, n_yes, "yes"), (1, n_no, "no"), (2, n_mid, "middle")):
        sub = d[d["goldstandard2"] == g]; idx = list(sub.index); random.Random(10 + g).shuffle(idx)
        for i in sorted(idx[:k]):
            r = d.loc[i]
            out.append(item(id=f"circa.{i}", phenomenon="indirect_answer", source="circa", subset="indirect", role="target",
                            context=f"Context: {r['context']}\nX asks: \"{r['question-X']}\"\nY answers: \"{r['answer-Y']}\"", question="What does Y's answer mean?",
                            options=["Yes", "No", "Neither yes nor no (in the middle)"], option_kinds=["yes", "no", "middle"], gold=g,
                            meta={"gold_class": nm, "judgements": str(r["judgements"]), "pool_after_filter": int(len(sub))}))
    write("circa", out)


def impl1k_items():
    d = pd.read_csv(RAW / "implicatures1k" / "implicatures.csv"); out = []
    for i, r in d.iterrows():
        out.append(item(id=f"impl1k.{i}", phenomenon="conversational_implicature", source="implicatures1k", subset="all", role="target",
                        context=f"{str(r['context']).strip()}\nReply: {str(r['response']).strip()}", question="Which statement does the reply implicate?",
                        options=[str(r["correct_implicature"]).strip(), str(r["incorrect_implicature"]).strip()], option_kinds=["implicature", "foil"], gold=0, meta={}))
    write("implicatures1k", out)


def nope_items():
    """NOPE (Parrish et al. 2021; HF tasksource/nope): naturally occurring presuppositions as NLI. eval + adversarial files (346 each, same uid and labels, different hypotheses), 3-way entail / neutral / contradict."""
    out = []; L = {"E": 0, "N": 1, "C": 2}
    OPT = ["Yes, the statement must be true if the passage is true.", "Not necessarily: the statement may be true or false.", "No, the statement must be false if the passage is true."]
    for fn in ("eval", "adv"):
        for r in map(json.loads, open(RAW / "nope" / f"nli_corpus.{fn}.jsonl")):
            md = r["metadata"]
            out.append(item(id=f"nope.{fn}.{r['uid']}", phenomenon="presupposition_projection", source="nope", subset=md["trigger_type"], role="target",
                            context=f"Passage: {r['premise']}", question=f"Assume the passage is true. Does this statement follow? \"{r['hypothesis']}\"", options=list(OPT),
                            option_kinds=["entailment", "neutral", "contradiction"], gold=L[r["label"]], meta={"uid": r["uid"], "uid_base": r["uid"].split("-")[0], "file": fn, "type": md.get("original_negated"), "adversarial": md.get("adversarial")}))
    write("nope", out)


def xl_items():
    """Non-English slice (machine-translated; see translate_items.py): the 160 IMPPRES quantifier/connective test pairs with the two sentences in German / Chinese (+ the English originals as a same-items control),
    English instruction / question / options (felicity framing). Gold and option order are unchanged; the translation is NOT human-verified."""
    base = {r["id"]: r for r in map(json.loads, open(OUT / "imppres_scalar_target.jsonl"))}
    out = []
    for lang in ("en", "de", "zh"):
        rows = [json.loads(l) for l in open(HERE / "translations" / f"imppres_{'de' if lang == 'en' else lang}.jsonl")]
        for r in rows:
            if not r["ok"]: continue
            if lang == "zh" and not (re.search("[\u4e00-\u9fff]", r["premise_t"]) and re.search("[\u4e00-\u9fff]", r["hypothesis_t"])): continue   # mechanical validity filter (lead 2026-10-05): drop zh pairs with no Chinese characters in either sentence
            it = dict(base[r["item_id"]]); fel = dict(it["variants"]["felicity"])
            prem, hyp = (it["meta"]["premise"], it["meta"]["hypothesis"]) if lang == "en" else (r["premise_t"], r["hypothesis_t"])
            it.update(fel); it["context"] = f'A speaker says: "{prem}"\nSecond statement: "{hyp}"'; it["variants"] = {}
            it["id"] = f"xl{lang}.{r['item_id']}"; it["source"] = "imppres_xl"; it["meta"] = {**it["meta"], "lang": lang, "orig_id": r["item_id"], "translator": r.get("model") if lang != "en" else None}
            out.append(it)
    write("imppres_xl", out)



# ----------------------------------------------------------------------------------------------------------------
# 13. METHOD sets (METHOD_DESIGN v1, pre-declared 2026-10-05): frozen DEV (selection only) and TEST item files + manifests + contract. Items carry the FINAL framing (variant applied), original gold,
#     meta.set = set name, meta.family in {speaker, presup, literal}. GSM8K goes to method_gsm_{dev,test}.jsonl (free numeric answers, own runner).
# ----------------------------------------------------------------------------------------------------------------
def method_items():
    sys.path.insert(0, str(HERE.parent / "harness")); from common import load_items_from_config
    def L(n): return [json.loads(l) for l in open(OUT / f"{n}.jsonl")]
    # items already inspected in E1/E8/E10/E6/E9 (core a3 items, the LUDWIG b2 items): they belong to DEV even if they sit in a test half
    insp = {}
    for cfgn in ("E1_main", "E8", "E10_seeds", "E6_prior", "E9_ruis"):
        for it in load_items_from_config(json.load(open(HERE.parent / "configs" / f"{cfgn}.json"))): insp.setdefault(it["bucket"], set()).add(it["id"])
    core_a3 = insp.get("a3_fel", set()) | insp.get("a3_nli", set()); core_b2 = {i for i in insp.get("b2_ludwig", set())}
    dev, test = [], []
    def add(split_of, base, setname, family, ctx=None, q=None, opts=None, kinds=None, gold=None, oid=None, extra=None):
        it = dict(base); oid = oid or base["id"]
        it.update(id=f"m1.{setname}.{oid}", source="method", subset=setname, role="target", phenomenon=family)
        if ctx is not None: it["context"] = ctx
        if q is not None: it["question"] = q
        if opts is not None: it.update(options=opts, option_kinds=kinds, gold=gold, literal=None)
        it["variants"] = {}; it["question_alt"] = None; it["meta"] = {**base.get("meta", {}), "set": setname, "family": family, "orig_id": oid, **(extra or {})}
        it["split"] = split_of; (dev if split_of == "dev" else test).append(it)
    def assign(base, core_ids):   # DEV = dev half + inspected items; TEST = test half minus inspected
        return "dev" if (base.get("split") == "dev" or base["id"] in core_ids) else "test"
    sc = L("imppres_scalar_target")
    for b in sc:
        if b["subset"] not in ("quantifiers", "connectives"): continue
        sp = assign(b, core_a3); tag = "quant" if b["subset"] == "quantifiers" else "conn"
        fel = b["variants"]["felicity"]
        add(sp, b, f"imppres_{tag}_fel", "speaker", ctx=fel["context"], q=fel["question"], opts=fel["options"], kinds=fel["option_kinds"], gold=fel["gold"])          # gold = pragmatic reading
        lg = str(b["meta"]["gold_label_log"]); gy = 0 if lg == "0" else 1
        add(sp, b, f"imppres_{tag}_truth", "literal", ctx=fel["context"], q="Does the second statement strictly follow, by logic alone, from the first? Answer Yes or No.",
            opts=["Yes", "No"], kinds=["entails", "does_not_entail"], gold=gy)                                                                                          # gold = literal (logical) reading
    for b in L("ruis2022_dev"): add("dev", b, "ludwig", "speaker")
    for b in L("ruis2022_test"):
        if b.get("split") == "dev": continue                           # the test file's own dev half is not used
        add("dev" if b["id"] in core_b2 else "test", b, "ludwig", "speaker")
    for b in L("circa"): add(b["split"], b, "circa", "speaker")
    for b in L("pub6"):
        if b["subset"] in ("task2", "task3", "task6"): add(b["split"], b, "pub_" + b["subset"], "speaker")
    for b in L("figqa"): add(b["split"], b, "figqa", "speaker")
    for b in L("implicaturex"):
        k = b["meta"]["kind"]
        if k == "rec": add(b["split"], b, "ix_rec", "speaker")
        elif k == "can": add(b["split"], b, "ix_can", "literal")
    for b in L("nope"): add(b["split"], b, "nope", "presup")
    rp = random.Random(5)
    for sp in ("dev", "test"):
        pool = [b for b in L("imppres_presupposition") if b["split"] == sp]; rp.shuffle(pool)
        for b in sorted(pool[:300 if sp == "test" else 200], key=lambda x: x["id"]): add(sp, b, "imppres_presup", "presup")
    fo = {"train": [json.loads(l) for l in open(RAW / "folio" / "folio_v2_train.jsonl")], "validation": [json.loads(l) for l in open(RAW / "folio" / "folio_v2_validation.jsonl")]}
    rf = random.Random(7); tr = list(fo["train"]); rf.shuffle(tr)
    LAB = {"True": 0, "False": 1, "Uncertain": 2}
    for sp, rows in (("dev", sorted(tr[:200], key=lambda r: str(r["example_id"]))), ("test", fo["validation"])):
        for r in rows:
            base = {"id": f"folio.{r['example_id']}", "meta": {"story_id": str(r["story_id"])}}
            add(sp, base, "folio", "literal", ctx=f"Premises:\n{r['premises'].strip()}\n\nConclusion: {r['conclusion'].strip()}", q="Based only on the premises, is the conclusion true, false, or uncertain?",
                opts=["True", "False", "Uncertain"], kinds=["true", "false", "uncertain"], gold=LAB[r["label"]], oid=str(r["example_id"]))
    for nm, rows in (("method_dev", dev), ("method_test", test)):
        with open(OUT / f"{nm}.jsonl", "w") as f:
            for it in rows: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    # GSM8K (free numeric): DEV 150 from train, TEST 500 from test (seeded); the gold is the number after '####'
    import re as _re
    gs = {}
    for sp, fn, n in (("dev", "train.parquet", 150), ("test", "test.parquet", 500)):
        d = pd.read_parquet(RAW / "gsm8k" / fn); idx = list(range(len(d))); random.Random(11).shuffle(idx); rows = []
        for i in sorted(idx[:n]):
            ans = d.iloc[i].answer.split("####")[-1].strip().replace(",", ""); rows.append({"id": f"m1.gsm8k.{sp}{i}", "problem": d.iloc[i].question, "answer": ans, "split": sp})
        with open(OUT / f"method_gsm_{sp}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        gs[sp] = [r["id"] for r in rows]
    # manifests (frozen) + contract
    mdir = HERE / "manifests"; mdir.mkdir(exist_ok=True); man = {}
    for nm, rows in (("DEV", dev), ("TEST", test)):
        by = {}
        for it in rows: by.setdefault(it["meta"]["set"], []).append(it["id"])
        by["gsm8k"] = gs["dev" if nm == "DEV" else "test"]
        man[nm] = by; blob = json.dumps(by, sort_keys=True)
        (mdir / f"METHOD_{nm}.json").write_text(json.dumps({"sets": by, "n": {k: len(v) for k, v in by.items()}, "sha256_16": hashlib.sha256(blob.encode()).hexdigest()[:16], "frozen": "2026-10-05"}, indent=1))
    fam = {}
    for it in dev + test: fam[it["meta"]["set"]] = it["meta"]["family"]
    fam["gsm8k"] = "literal"
    rows_md = ["| set | family | gold meaning | DEV n | TEST n |", "|---|---|---|---|---|"]
    GM = {"speaker": "original gold = the speaker-intended / pragmatic reading", "presup": "original NLI gold (presupposition projection)", "literal": "literal / logical gold (no enrichment)"}
    for k in sorted(fam): rows_md.append(f"| {k} | {fam[k]} | {GM[fam[k]]} | {len(man['DEV'].get(k, []))} | {len(man['TEST'].get(k, []))} |")
    (mdir / "METHOD_CONTRACT.md").write_text("# METHOD data contract (frozen 2026-10-05, before any test run)\nOriginal gold labels and task semantics throughout. DEV = dev halves + every item inspected in E1/E8/E10/E6/E9 (selection only); TEST = fresh held-out items. IMPPRES some/or: felicity framing gold = pragmatic reading; TRUTH framing ('does the second statement strictly follow, by logic alone?') gold = literal (logical) reading from IMPPRES gold_label_log; ImplicatureX rec = implicature holds, can = cancelled (gold literal); FOLIO v2 (tasksource/folio): DEV = 200 seeded train items, TEST = validation 203; GSM8K (openai/gsm8k, MIT): DEV 150 train items, TEST 500 test items, numeric answer after ####.\n\n" + "\n".join(rows_md) + "\n")
    print("method DEV", len(dev), "TEST", len(test), {k: len(v) for k, v in man["TEST"].items()})



# ----------------------------------------------------------------------------------------------------------------
# 14. CONTEXT-REVERSAL set (FRESH; triage amendment 2026-10-05): 60 NEW scenario pairs from the A1-ctx generator (new names, nouns and properties, none of the 50 A1-ctx scenarios): the same "Some of the X are P." under FULL access
#     (speaker checked all: the description is misleading, pragmatic gold = No) and PARTIAL access (speaker checked only two: the description is fine, literal gold = Yes). Metric = PAIR accuracy (both members right), keyed by meta.pair.
# ----------------------------------------------------------------------------------------------------------------
def ctxrev_items():
    names = ["Alba", "Bruno", "Chloe", "Dev", "Elena", "Felix", "Gita", "Hiro", "Isla", "Jonas", "Kara", "Luca", "Mina", "Nico", "Orla", "Pablo", "Rhea", "Stella", "Teo", "Ulla", "Vik", "Willa", "Yusuf", "Zara", "Amir", "Bea", "Cyrus", "Dalia", "Emil", "Fiona",
             "Gus", "Hedda", "Idris", "Jana", "Karl", "Liv", "Mateo", "Nell", "Otis", "Pia", "Remy", "Sana", "Toby", "Una", "Viggo", "Wanda", "Xena", "Yuri", "Zeno", "Anya", "Boris", "Cleo", "Dario", "Edda", "Finn", "Gwen", "Hal", "Ines", "Joel", "Kaia"]
    nouns = [("scarves", "scarf"), ("mugs", "mug"), ("sofas", "sofa"), ("kites", "kite"), ("ropes", "rope"), ("towels", "towel"), ("shovels", "shovel"), ("ladders", "ladder"), ("drums", "drum"), ("flutes", "flute"), ("tents", "tent"), ("stools", "stool"),
             ("baskets", "basket"), ("mirrors", "mirror"), ("coats", "coat"), ("hats", "hat"), ("belts", "belt"), ("jars", "jar"), ("kettles", "kettle"), ("fans", "fan"), ("trays", "tray"), ("vases", "vase"), ("rugs", "rug"), ("blankets", "blanket"),
             ("lanterns", "lantern"), ("brushes", "brush"), ("cameras", "camera"), ("speakers", "speaker"), ("drones", "drone"), ("tablets", "tablet"), ("keyboards", "keyboard"), ("monitors", "monitor"), ("routers", "router"), ("cables", "cable"),
             ("hammers", "hammer"), ("drills", "drill"), ("saws", "saw"), ("buckets", "bucket"), ("kayaks", "kayak"), ("scooters", "scooter"), ("tractors", "tractor"), ("vans", "van"), ("boots", "boot"), ("sandals", "sandal"), ("jackets", "jacket"),
             ("aprons", "apron"), ("cushions", "cushion"), ("pillows", "pillow"), ("notebooks", "notebook"), ("folders", "folder"), ("maps", "map"), ("posters", "poster"), ("stamps", "stamp"), ("coins", "coin"), ("medals", "medal"), ("trophies", "trophy"),
             ("pans", "pan"), ("gates", "gate"), ("fences", "fence"), ("whistles", "whistle"), ("ribbons", "ribbon")]
    props = [("are dented", "was dented", "were dented"), ("are faded", "was faded", "were faded"), ("are stained", "was stained", "were stained"), ("are bent", "was bent", "were bent"), ("are wet", "was wet", "were wet"),
             ("are cold", "was cold", "were cold"), ("are sealed", "was sealed", "were sealed"), ("are labelled", "was labelled", "were labelled"), ("are scratched", "was scratched", "were scratched"), ("are dusty", "was dusty", "were dusty"),
             ("are patched", "was patched", "were patched"), ("are sticky", "was sticky", "were sticky"), ("are frayed", "was frayed", "were frayed"), ("are rusty", "was rusty", "were rusty"), ("are wrinkled", "was wrinkled", "were wrinkled"),
             ("are tangled", "was tangled", "were tangled"), ("are folded", "was folded", "were folded"), ("are warped", "was warped", "were warped"), ("are soaked", "was soaked", "were soaked"), ("are tagged", "was tagged", "were tagged"),
             ("are expired", "was expired", "were expired"), ("are noisy", "was noisy", "were noisy"), ("are shiny", "was shiny", "were shiny"), ("are scorched", "was scorched", "were scorched"), ("are jammed", "was jammed", "were jammed"),
             ("are snapped", "was snapped", "were snapped"), ("are numbered", "was numbered", "were numbered"), ("are loose", "was loose", "were loose"), ("are stuck", "was stuck", "were stuck"), ("are sharp", "was sharp", "were sharp")]
    where = ["in closed boxes in a storage room", "in sealed bags in a van", "in a dark shed behind a barn", "in locked cabinets in a school", "in a heap under a tarpaulin", "in covered crates at a harbour", "in the lockers of a gym",
             "in a warehouse with no windows", "in boxes stacked in an attic", "in drawers of a tall chest", "in a basement behind a locked door", "on shelves in a dim garage", "in a trunk at the back of a theatre", "in sacks on a market stall at night"]
    rng = random.Random(2026); rng.shuffle(nouns); rng.shuffle(props); rng.shuffle(where); seen = set(); nn = []
    for t in nouns:
        if t[0] not in seen: seen.add(t[0]); nn.append(t)
    A_used = {t[2] for t in A1CTX}; A_used_p = {t[5] for t in A1CTX}; A_used_n = {t[0] for t in A1CTX}
    nn = [t for t in nn if t[0] not in A_used]; props = [t for t in props if t[0] not in A_used_p]; names = [n for n in names if n not in A_used_n]
    N = 60; assert len(nn) >= N and len(names) >= N, (len(nn), len(names))
    NUMS = [3, 4, 5, 6, 8]
    out = []
    for k in range(N):
        pl, sg = nn[k]; pres, pst_sg, pst_pl = props[k % len(props)]; name = names[k]; n = NUMS[k % 5]; wh = where[k % len(where)]; Nw = NUMW[n]
        setting = f"There are {Nw} {pl} {wh}."; utt = f"Some of the {pl} {pres}."
        access = {"full": f"{name} checks all {Nw} of the {pl}.", "partial": f"{name} has time to check only two of the {pl}; the other {NUMW[n - 2]} {'is' if n - 2 == 1 else 'are'} not checked."}
        obs = {"full": f"Every {sg} that {name} checked {pst_sg}.", "partial": f"Both {pl} that {name} checked {pst_pl}."}
        q = f'Given what {name} found out, is "{utt[:-1]}" a good way for {name} to describe the {Nw} {pl}? Answer Yes or No.'
        for acc in ("full", "partial"):
            full = acc == "full"
            out.append(item(id=f"m1.ctxrev.{k}.{acc}", phenomenon="speaker_knowledge", source="method", subset="ctxrev", role="target", context=f'{setting} {access[acc]} {obs[acc]} {name} says: "{utt}"', question=q,
                            options=YESNO, option_kinds=["literal", "pragmatic"] if full else ["yes", "no"], gold=1 if full else 0, literal=0,
                            meta={"set": "ctxrev", "family": "speaker", "pair": k, "access": acc, "fresh": True, "orig_id": f"{k}.{acc}", "name": name, "noun": pl, "property": pres, "n": n}))
    for it in out: it["split"] = "test"; it["variants"] = {}; it["question_alt"] = None
    with open(OUT / "method_ctxrev.jsonl", "w") as f:
        for it in out: f.write(json.dumps(it, ensure_ascii=False) + "\n")
    mf = HERE / "manifests" / "METHOD_TEST.json"; m = json.loads(mf.read_text()); m["sets"]["ctxrev"] = [it["id"] for it in out]; m["n"]["ctxrev"] = len(out)
    m["fresh_sets"] = ["ctxrev"]; m["sha256_16"] = hashlib.sha256(json.dumps(m["sets"], sort_keys=True).encode()).hexdigest()[:16]; mf.write_text(json.dumps(m, indent=1))
    print("ctxrev pairs", N, "items", len(out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ext":      # add the expansion sets without touching the existing files / manifest entries
        manifest.update(json.loads((HERE / "manifest.json").read_text())); circa_items(); impl1k_items(); nope_items(); xl_items(); method_items(); ctxrev_items()
        (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True)); sys.exit(0)
    imppres_scalar(); imppres_presup(); bn_items(); a1ctx_items(); a2_items(); ix_items(); hu_items(); ruis_items(); pub_items(); figqa_items(); pub6_items()
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print("wrote manifest.json with", len(manifest), "files")
