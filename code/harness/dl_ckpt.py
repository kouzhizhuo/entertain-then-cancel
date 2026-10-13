#!/usr/bin/env python
"""Download one checkpoint for E2 (hf-mirror, one process per repo). usage: dl_ckpt.py <name> <repo> [revision]  -> $R/models/E2/<name>/ (+ .dl_ok or .dl_fail).
Only officially ungated repos: a 401/403 marks .dl_fail (skip, tell the lead). Nothing is ever downloaded from unofficial re-uploads."""
import os, sys, time
from pathlib import Path
name, repo = sys.argv[1], sys.argv[2]; rev = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
d = Path(os.environ.get("P2_DEST", os.environ.get("P2_R", "<path>") + "/models/E2")) / name; d.mkdir(parents=True, exist_ok=True)
if (d / ".dl_ok").exists(): print("already ok", name); sys.exit(0)
os.environ.update(HF_ENDPOINT="https://hf-mirror.com", HF_HUB_DISABLE_XET="1", HF_HUB_DOWNLOAD_TIMEOUT="30", HF_HUB_ETAG_TIMEOUT="30")
from huggingface_hub import snapshot_download
for k in range(4):
    try:
        snapshot_download(repo, revision=rev, local_dir=str(d), max_workers=8, ignore_patterns=["original/*", "metal/*", "*.gguf", "*.msgpack", "*.h5", "*.onnx"])
        import json
        cf = d / "config.json"
        try:
            cfg = json.load(open(cf))
        except Exception:
            cfg = {}
        if cfg.get("model_type") == "olmo2-retrofit":   # intermediate RL-Zero steps ship a custom class name; same tensors as Olmo3 if the key sets are identical
            from huggingface_hub import hf_hub_download
            ref = json.load(open(hf_hub_download("allenai/Olmo-3-7B-RL-Zero-IF", "model.safetensors.index.json")))["weight_map"].keys()
            mine = json.load(open(d / "model.safetensors.index.json"))["weight_map"].keys() if (d / "model.safetensors.index.json").exists() else []
            if set(ref) == set(mine):
                json.dump(cfg, open(d / "config.json.orig", "w"))
                cfg.update({"model_type": "olmo3", "architectures": ["Olmo3ForCausalLM"]}); json.dump(cfg, open(cf, "w"), indent=2)
                (d / ".config_patched").write_text("olmo2-retrofit -> olmo3; weight_map keys identical to allenai/Olmo-3-7B-RL-Zero-IF (config compat patch, UNVERIFIED equivalence)\n")
            else:
                (d / ".dl_fail").write_text("olmo2-retrofit checkpoint with weight keys different from Olmo3: not patched\n"); print("FAIL (retrofit keys differ)", name, flush=True); sys.exit(0)
        (d / ".dl_ok").touch(); print("OK", name, repo, rev, flush=True); sys.exit(0)
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:160]}"; print("RETRY", k, msg, flush=True)
        if any(x in msg for x in ("401", "403", "Gated", "gated", "Repository Not Found", "Revision Not Found", "404")): break
        time.sleep(20)
(d / ".dl_fail").touch(); print("FAIL", name, repo, rev, flush=True)
