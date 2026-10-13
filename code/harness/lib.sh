# shared helpers for tasks.d/*.sh (sourced by p2_taskd.sh). GPU 4 on <host>.
R=${P2_R:-<path> P=$R/runs/P2; EXP=$P/exp; LOG=$P/queue.log
log(){ echo "[$(date '+%F %T')] $*" >> $LOG; }
mp(){ case "$1" in /*) echo "$1";; *) echo "$R/models/$1";; esac; }
ready(){ d=$(mp $1); [ -d $d ] && { case $d in */models/E2/*|*/models/B/*) [ -f $d/.dl_ok ];; *) ls $d/*.safetensors > /dev/null 2>&1;; esac; } && { [ ! -f $R/logs/dl_P2_$(basename $1).log ] || grep -aq "^OK" $R/logs/dl_P2_$(basename $1).log; }; }
# wait_model: prefetched/downloaded checkpoints (models/E2/*) may take hours; anything else must already be on disk (P3 deletes its weights after use) -> fail after ~3 min
wait_model(){ n=6; case "$(mp $1)" in */models/E2/*|*/models/B/*) n=1440;; esac
  for i in $(seq 1 $n); do ready $1 && return 0; sleep 30; done; log "model $1 not ready/missing weights (waited $((n*30))s)"; return 1; }
# vrun <config> <model dir name or path> <out group> <out name> [extra run_vllm args]
vrun(){ cfg=$1; m=$2; grp=$3; nm=$4; shift 4
  [ -f $P/$grp/$nm/results.json ] && { log "skip $cfg $nm"; return 0; }
  wait_model $m || return 1; gate
  log "run $cfg $nm"; (cd $EXP/harness && python run_vllm.py --config ../configs/$cfg.json --model $(mp $m) --out $P/$grp/$nm "$@" > $P/${grp}_$nm.log 2>&1); log "end $cfg $nm rc=$?"; }
lprun(){ m=$1; nm=$2; [ -f $P/E2_lp/$nm/results.json ] && { log "skip lp $nm"; return 0; }; wait_model $m || return 1; gate
  log "run lp $nm"; (cd $EXP/harness && python run_lp.py --config ../configs/E2_lp.json --model $(mp $m) --out $P/E2_lp/$nm > $P/E2_lp_$nm.log 2>&1); log "end lp $nm rc=$?"; }
xp(){ python $EXP/analysis/export_e1.py $1 $P $EXP/results/$2 >> $LOG 2>&1; log "exported $1 -> results/$2"; }

# dlm <repo> <name> [revision]: synchronous download to $R/models/<name> (mirror, official repos only), .dl_ok / .dl_fail marker
dlm(){ [ -f $R/models/$2/.dl_ok ] && return 0; log "download $1 -> $2"; P2_DEST=$R/models HF_HUB_OFFLINE=0 python $EXP/harness/dl_ckpt.py $2 $1 ${3:--} >> $P/dl_lane.log 2>&1; [ -f $R/models/$2/.dl_ok ]; }
# gate: on shared hosts wait while another user's process sits on this lane's GPU (P2_GATE=1) or a HOLD file exists
busy_other(){ nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | while IFS=', ' read -r u p; do
    idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$u" | cut -d, -f1)
    [ "$idx" = "$CUDA_VISIBLE_DEVICES" ] && { o=$(ps -o user= -p $p 2>/dev/null | tr -d ' '); [ -n "$o" ] && [ "$o" != "$(whoami)" ] && echo busy; }; done | grep -q busy; }
gate(){ [ -z "$P2_GATE" ] && return 0; while [ -f $P/HOLD ] || busy_other; do log "gate: HOLD or another user on GPU $CUDA_VISIBLE_DEVICES"; sleep 120; done; }
# lprun2 <config> <model dir/path> <run name> <out group>: P-lp with an arbitrary config (E9 Hu protocol on B1)
lprun2(){ cfg=$1; m=$2; nm=$3; grp=$4; [ -f $P/$grp/$nm/results.json ] && { log "skip lp2 $grp $nm"; return 0; }; wait_model $m || return 1; gate
  log "run lp2 $cfg $nm"; (cd $EXP/harness && python run_lp.py --config ../configs/$cfg.json --model $(mp $m) --out $P/$grp/$nm > $P/${grp}_$nm.log 2>&1); log "end lp2 $cfg $nm rc=$?"; }
