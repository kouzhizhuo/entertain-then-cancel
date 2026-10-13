#!/bin/bash
# Background prefetch of the E2 manifest (network/CPU only, no GPU): 3 parallel downloads via hf-mirror, official repos only. Idempotent (skips .dl_ok).
A=${P2_R:-<path> P=$A/runs/P2; EXP=$P/exp; source $A/envs/gen/bin/activate
cat $EXP/harness/tasks25_gpu7/e2_manifest.tsv | xargs -P3 -L1 bash -c 'HF_HUB_OFFLINE=0 P2_R='$A' python '$EXP'/harness/dl_ckpt.py "$0" "$1" "$2" >> '$P'/dl_E2.log 2>&1'
echo "[$(date '+%F %T')] prefetch finished" >> $P/queue.log
