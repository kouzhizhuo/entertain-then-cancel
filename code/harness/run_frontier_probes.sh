#!/bin/bash
# E1F yes-bias discriminators (lead, METHOD_CHECK s2): no-utterance prior probe and polarity-flipped question on the 160 a3-quant/conn items, 2 samples, lowest vs highest dose. Starts after run_frontier_s2.sh. Every call -> logs/api_spend.md.
cd "$(dirname "$0")"; L=../results/api_logs/probes.log
until grep -q ALLDONE ../results/api_logs/s2.log; do sleep 30; done
r(){ python3 run_api.py run --model $1 --config ../configs/$2.json --doses $3 --out_root $4 --cap_usd 12 --global_cap 150 >> $L 2>&1; }
for fmt in prior flip; do
 ( r openai/gpt-5 E1f_$fmt "eff-min eff-high" api_$fmt ) &
 ( r anthropic/claude-sonnet-4.6 E1f_$fmt "off512 bud16k" api_$fmt; r anthropic/claude-opus-4.8 E1f_$fmt "off512 eff-high" api_$fmt ) &
 wait
done; echo ALLDONE >> $L
