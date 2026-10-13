#!/bin/bash
# E1F second-format check (lead request): 3-way NLI forced choice, same a3-quant/a3-conn items, lowest vs highest dose. Every call goes to logs/api_spend.md.
cd "$(dirname "$0")"; export OPENAI_API_KEY   # key comes from the environment; never printed
for spec in "openai/gpt-5|eff-min eff-high" "anthropic/claude-sonnet-4.6|off bud16k" "anthropic/claude-opus-4.8|off eff-high"; do
  m=${spec%%|*}; d=${spec##*|}
  python3 run_api.py run --config ../configs/E1f_mc.json --model $m --doses $d --out_root api_mc --cap_usd 8 --global_cap 150 >> ../results/api_logs/mc.log 2>&1
done
echo ALLDONE >> ../results/api_logs/mc.log
