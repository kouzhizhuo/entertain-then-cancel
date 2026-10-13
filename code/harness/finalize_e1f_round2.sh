#!/bin/bash
# after the round-2 API runs: fill dropped calls (resumable), rebuild E1F, then the artefact and probe analyses
cd "$(dirname "$0")"; X=..; L=$X/results/api_logs
until grep -q ALLDONE $L/probes.log 2>/dev/null; do sleep 30; done
for k in 1 2; do bash run_frontier_s2.sh; done          # second pass only runs calls that are still missing
cp $L/probes.log $L/probes.log.1; sed -i.bak '/ALLDONE/d' $L/probes.log; rm -f $L/probes.log.bak
bash run_frontier_probes.sh
cd $X/..; python3 exp/analysis/api_finalize.py > $L/finalize2.log 2>&1
python3 exp/analysis/export_e1.py e1fr exp/results/api_runs exp/results/E1F >> $L/finalize2.log 2>&1
python3 exp/analysis/e1f_artefact.py >> $L/finalize2.log 2>&1
python3 exp/analysis/e1f_probes.py >> $L/finalize2.log 2>&1
echo FINAL_DONE >> $L/finalize2.log
