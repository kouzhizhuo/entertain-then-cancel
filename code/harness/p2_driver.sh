#!/bin/bash
# P2 unattended driver (<host>, GPU 4). Resumable: every step is skipped when its output exists. Log: runs/P2/queue.log.
# Order: E1 main (4B/8B/14B) -> gold pool + audit files -> A1-ctx main -> STRAT -> ladder (1.7B,0.6B,32B-FP8,30B-A3B-FP8 + MATH-500) -> MATH-500 4B/8B/14B -> gpt-oss-20b E3.
# After each block it runs analysis/export_e1.py -> exp/results/<block>/{summary.json,summary_all.json,DONE}. Restart-safe: `bash p2_driver.sh`.
R=<path> P=$R/runs/P2; EXP=$P/exp; LOG=$P/queue.log
source $R/envs/gen/bin/activate; cd $EXP/harness
export CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }
exec 9>$P/driver.lock; flock -n 9 || { log "another driver holds the lock; exiting"; exit 0; }
log "driver start (pid $$)"
# hand-over from the interim master queue at a clean boundary (after E1_main/Qwen3-8B is finished)
for i in $(seq 1 720); do
  [ -f $P/E1_main/Qwen3-8B/results.json ] && break
  pgrep -f "bash .*p2_master.sh" > /dev/null || break
  sleep 15
done
MP=$(pgrep -f "bash .*p2_master.sh" | head -1)
if [ -n "$MP" ]; then G=$(ps -o pgid= -p $MP | tr -d ' '); log "stopping interim master group $G"; kill -TERM -- -$G; sleep 20; kill -KILL -- -$G 2>/dev/null; fi
for i in $(seq 1 24); do u=$(nvidia-smi --id=4 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 3000 ] && break; sleep 10; done
ready(){ [ -d $R/models/$1 ] && ls $R/models/$1/*.safetensors > /dev/null 2>&1 && { [ ! -f $R/logs/dl_P2_$1.log ] || grep -aq "^OK" $R/logs/dl_P2_$1.log; }; }
wait_model(){ for i in $(seq 1 1440); do ready $1 && return 0; sleep 30; done; log "model $1 never became ready"; return 1; }
vrun(){ cfg=$1; m=$2; out=$3; shift 3
  [ -f $P/$out/$m/results.json ] && { log "skip $cfg $m (done)"; return 0; }
  wait_model $m || return 1
  log "run $cfg $m"; python run_vllm.py --config ../configs/$cfg.json --model $R/models/$m --out $P/$out/$m "$@" > $P/${out}_$m.log 2>&1; log "end $cfg $m rc=$?"; }
vmath(){ m=$1; [ -f $P/E1s_math/$m/math500.json ] && { log "skip math $m"; return 0; }; wait_model $m || return 1
  log "run math $m"; python run_math.py --model $R/models/$m --out $P/E1s_math/$m > $P/E1s_math_$m.log 2>&1; log "end math $m rc=$?"; }
xp(){ mode=$1; dst=$2; python $EXP/analysis/export_e1.py $mode $P $EXP/results/$dst >> $LOG 2>&1; log "exported $mode -> results/$dst"; }

for m in Qwen3-8B Qwen3-4B Qwen3-14B; do vrun E1_main $m E1_main; done; xp e1 E1
# gold pool (due 2026-10-01 12:00): pool from Qwen3-8B, rater B = Qwen3-32B-FP8 non-thinking, audit file on CPU
mkdir -p $EXP/gold
[ -f $EXP/gold/reference_pool.jsonl ] || { log "gold pool"; python gold_pool.py pool --model $R/models/Qwen3-8B --out $EXP/gold > $P/gold_pool.log 2>&1; log "gold pool rc=$?"; }
[ -f $EXP/gold/labels_B.jsonl ] || { wait_model Qwen3-32B-FP8; log "gold labels B"; python gold_pool.py label --model $R/models/Qwen3-32B-FP8 --out $EXP/gold > $P/gold_label.log 2>&1; log "gold labels rc=$?"; }
[ -f $EXP/gold/HUMAN_AUDIT_TODO.jsonl ] || { python gold_pool.py audit --out $EXP/gold >> $LOG 2>&1; log "audit file written"; }
for m in Qwen3-8B Qwen3-4B Qwen3-14B; do vrun a1ctx_main $m a1ctx_main; done; xp a1ctx A1CTX
for m in Qwen3-8B Qwen3-4B Qwen3-14B; do vrun E1_strat $m E1_strat; done
for m in Qwen3-1.7B Qwen3-0.6B Qwen3-32B-FP8 Qwen3-30B-A3B-FP8; do vrun E1s_ladder $m E1s; vmath $m; xp e1s E1S; done
for m in Qwen3-4B Qwen3-8B Qwen3-14B; do vmath $m; done; xp e1s E1S
# E3: gpt-oss-20b effort-control check (50 dev items) then the core sets at low/medium/high
if ready gpt-oss-20b; then
  [ -f $P/E3_check/gpt-oss-20b/results.json ] || { log "run E3_check"; python run_gptoss.py --config ../configs/E3_check.json --model $R/models/gpt-oss-20b --out $P/E3_check/gpt-oss-20b > $P/E3_check.log 2>&1; log "end E3_check rc=$?"; }
  [ -f $P/E3_gptoss/gpt-oss-20b/results.json ] || { log "run E3_gptoss"; python run_gptoss.py --config ../configs/E3_gptoss.json --model $R/models/gpt-oss-20b --out $P/E3_gptoss/gpt-oss-20b > $P/E3_gptoss.log 2>&1; log "end E3_gptoss rc=$?"; }
  xp e3 E3
fi
echo ALLDONE > $P/driver.done; log "driver finished"
