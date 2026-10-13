#!/bin/bash
# P2 task-directory driver (<host> GPU 4): runs tasks.d/*.sh in lexical order, each once (markers .done/.fail), polls for new tasks; STOP file ends it.
# Add work by dropping a NN_name.sh into exp/harness/tasks.d/ (it may `source ../lib.sh`). Restart-safe. Log: runs/P2/queue.log
export P2_R=${P2_R:-<path> R=$P2_R; P=$R/runs/P2; EXP=$P/exp; D=${P2_TASKS:-$EXP/harness/tasks.d}; GPU=${P2_GPU:-4}
source $R/envs/gen/bin/activate; export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
source $EXP/harness/lib.sh
exec 9>$P/taskd_gpu$GPU.lock; flock -n 9 || { log "taskd already running"; exit 0; }
log "taskd start gpu=$GPU tasks=$D (pid $$)"
while [ ! -f $P/STOP_gpu$GPU ] && [ ! -f $P/STOP ]; do
  ran=0
  for t in $(ls $D/*.sh 2>/dev/null | sort); do
    { [ -f $t.done ] || [ -f $t.fail ]; } && continue
    log "TASK start $(basename $t)"; ran=1
    if bash -c "export P2_R=$P2_R P2_GATE=$P2_GATE; source $EXP/harness/lib.sh; cd $EXP/harness; source $t" >> $P/taskd_$(basename $t).log 2>&1; then touch $t.done; log "TASK done $(basename $t)"; else touch $t.fail; log "TASK FAILED $(basename $t)"; fi
    break   # rescan from the top after every task: a lower-numbered (priority) task dropped in meanwhile runs next
  done
  [ $ran = 0 ] && sleep 120
done
log "taskd stopped (STOP file)"
