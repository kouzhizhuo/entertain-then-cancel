#!/usr/bin/env python3
"""R1-Distill-Llama tokenizer fix for transformers 5.x: tokenizer_config.json says LlamaTokenizerFast, which decodes byte-level BPE as 'ĊOkay,Ġso' (garbled text, ~100% invalid answers).
Sets tokenizer_class=PreTrainedTokenizerFast (keeps tokenizer_config.json.orig) and asserts a decode round-trip has no 'Ġ'/'Ċ'. usage: python fix_llama_tok.py <model_dir>"""
import json, shutil, sys
from pathlib import Path
d = Path(sys.argv[1]); p = d / "tokenizer_config.json"
if not (d / "tokenizer_config.json.orig").exists(): shutil.copy(p, d / "tokenizer_config.json.orig")
c = json.load(open(p)); c["tokenizer_class"] = "PreTrainedTokenizerFast"; json.dump(c, open(p, "w"), indent=2)
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(str(d)); s = "Okay, so I have this sentence.\n"
out = t.decode(t.encode(s, add_special_tokens=False)); assert "Ġ" not in out and "Ċ" not in out and out.strip() == s.strip(), repr(out)
print("tokenizer ok:", type(t).__name__, repr(out))
