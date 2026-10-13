#!/bin/bash
# graceful restart of the <host> GPU-4 task driver after its current task: STOP_gpu4 makes the old loop exit, then the new script (rescan after every task) starts
R=<path> P=$R/runs/P2; EXP=$P/exp; OLD=$(pgrep -f "bash .*p2_taskd.sh" | head -1)
touch $P/STOP_gpu4; echo "[$(date '+%F %T')] restart waiter: waiting for old taskd $OLD" >> $P/queue.log
while [ -n "$OLD" ] && kill -0 $OLD 2>/dev/null; do sleep 15; done
rm -f $P/STOP_gpu4; setsid nohup bash $EXP/harness/p2_taskd.sh > /dev/null 2>&1 < /dev/null &
echo "[$(date '+%F %T')] restart waiter: new taskd launched" >> $P/queue.log
