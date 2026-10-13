#!/bin/bash
# E1F robustness (coordinator/lead request): (i) Claude off baseline with a 512-token answer cap ('off512', same extraction), both formats, seeds 0+1; (ii) a second sample (seed 1, new option order) for GPT-5, Sonnet 4.6 and Opus 4.8 in both formats. Every call -> logs/api_spend.md.
cd "$(dirname "$0")"; L=../results/api_logs/s2.log
r(){ python3 run_api.py run --model $1 --config ../configs/$2.json --doses $3 ${4:+--out_root $4} --cap_usd 12 --global_cap 150 >> $L 2>&1; }
( for m in anthropic/claude-sonnet-4.6 anthropic/claude-opus-4.8; do
    r $m E1f_items_s01 off512; r $m E1f_mc_s01 off512 api_mc; done
  r anthropic/claude-sonnet-4.6 E1f_items_s1 "off bud1k bud4k bud16k"; r anthropic/claude-opus-4.8 E1f_items_s1 "off eff-low eff-high"
  r anthropic/claude-sonnet-4.6 E1f_mc_s1 "off bud16k" api_mc; r anthropic/claude-opus-4.8 E1f_mc_s1 "off eff-high" api_mc; echo ANTH_DONE >> $L ) &
( r openai/gpt-5 E1f_items_s1 "eff-min eff-low eff-med eff-high"; r openai/gpt-5 E1f_mc_s1 "eff-min eff-high" api_mc; echo OAI_DONE >> $L ) &
wait; echo ALLDONE >> $L
