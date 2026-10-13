#!/bin/bash
# DeepSeek-V4 as API SUBJECTS (user request): self-hosted endpoints, thinking off vs on via chat_template_kwargs; both answer formats (Yes/No inference and 3-way NLI), 2 samples. Every call -> logs/api_spend.md.
cd "$(dirname "$0")"; L=../results/api_logs/ds.log
for m in selfhosted/deepseek/deepseek-v4-flash-0731 selfhosted/deepseek/deepseek-v4-pro; do
  python3 run_api.py run --model $m --config ../configs/E1f_items_s01.json --doses off think --cap_usd 10 --global_cap 150 >> $L 2>&1
  python3 run_api.py run --model $m --config ../configs/E1f_mc_s01.json --doses off think --out_root api_mc --cap_usd 10 --global_cap 150 >> $L 2>&1
done; echo ALLDONE >> $L
