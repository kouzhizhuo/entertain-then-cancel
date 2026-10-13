#!/bin/bash
# Frontier panel (closed models as SUBJECTS) on the 340 test items; three provider streams in parallel; resumable; global spend cap $150 (first pass), logs/api_spend.md.
cd "$(dirname "$0")"; C=../configs/E1f_items.json; L=../results/api_logs; mkdir -p $L
( python3 run_api.py run --config $C --model openai/gpt-5 --doses eff-min eff-low eff-med eff-high; python3 run_api.py run --config $C --model openai/gpt-5.4 --doses eff-min eff-low eff-high; python3 run_api.py run --config $C --model openai/o4-mini --doses eff-low eff-med eff-high ) > $L/openai.log 2>&1 &
( python3 run_api.py run --config $C --model anthropic/claude-sonnet-4.6 --doses off bud1k bud4k bud16k; python3 run_api.py run --config $C --model anthropic/claude-opus-4.8 --doses off eff-low eff-high ) > $L/anthropic.log 2>&1 &
( python3 run_api.py run --config $C --model google/gemini-2.5-flash --doses bud0 bud1k bud8k; python3 run_api.py run --config $C --model google/gemini-2.5-pro --doses bud128 bud1k bud8k ) > $L/google.log 2>&1 &
wait; echo FRONTIER_DONE > $L/done
