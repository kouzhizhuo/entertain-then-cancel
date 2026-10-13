#!/bin/bash
# Pull finished P2 results to the Mac (one rsync per directory; --inplace for exFAT; traces.jsonl stays on the cluster). usage: bash exp/analysis/harvest.sh
X=<path> D30=<host>:<path> D25=<host>:<path> SL=slogin:<path>
rs(){ mkdir -p "$2"; for i in 1 2 3; do rsync -rt --inplace --exclude '._*' --exclude traces.jsonl -e "ssh -o BatchMode=yes -o ConnectTimeout=30" "$1/" "$2/" 2>&1 | grep -q -i "closed\|error" && sleep 30 || break; done; }
for g in a1ctx_main E1s_nt E1_seeds2 E8 E6 E2_lp E2_think E2_cot E1s_b0 E1_strat E3_gptoss_s2 E1s E1s_math E3_gptoss E3_check E1_main E7 A6_ix E9_ruis E9_hu_lp E4_load E1_scalar_full E1_pci_big E1_extra E1p_pop E9_grice E1q E7_full E1_probe E1R1_core E1R1_sfx M1_test M1_test_gsm E1_ds E1_dsx E1_xl E1_sfx E1_gptoss_ds E1_gptoss_sf E1_pangu E10_seeds; do
  for m in $(ssh -o BatchMode=yes <host> "ls <path> 2>/dev/null | grep -v -e bad_tokenizer -e BADTOK"); do [ -f $X/results/runs/$g/$m/results.json ] && continue; rs $D30/$g/$m $X/results/runs/$g/$m; done; done
for g in E8 E1s_nt E1_seeds2 E2_lp E1_extra E1_scalar_full E4_load E9_ruis E1p_pop E1b E1q E1_probe E1_ds E1_dsx E1_xl E1_sfx E1_gptoss_ds E1b_s2 E1q_s2 E1_pangu E10_seeds E7 E7_full E1R1_core E1R1_sfx; do
  for m in $(ssh -o BatchMode=yes <host> "ls <path> 2>/dev/null | grep -v -e bad_tokenizer -e BADTOK"); do [ -f $X/results/<host>/$g/$m/results.json ] && continue; rs $D25/$g/$m $X/results/<host>/$g/$m; done; done
for m in gpt-oss-120b Olmo-3-32B-Think; do [ -f $X/results/slurm3/a1ctx_main/$m/results.json ] || rs $SL/a1ctx_main/$m $X/results/slurm3/a1ctx_main/$m; done
for gm in "E1s_s2 Olmo-3-32B-Think" "E3_gptoss_s2 gpt-oss-120b" "E1p_pop DeepSeek-R1-Distill-Llama-70B" "E1b QwQ-32B" "E1b Llama-3.3-Nemotron-Super-49B-v1"; do set -- $gm; [ -f $X/results/slurm4/$1/$2/results.json ] || rs $SL/$1/$2 $X/results/slurm4/$1/$2; done   # job 406786: seeds 3-5; the 70B R1 rerun and the QwQ / Nemotron-Super runs (Slurm 407605 / 408986) are harvested now
rs $D30/backbones $X/results/backbones/<host>; rs $D25/backbones $X/results/backbones/<host>
rs <host>:<path> $X/gold_<host>_incoming; for f in labels_judge_gptoss20b.jsonl labels_judge_gptoss120b.jsonl; do [ -f $X/gold_<host>_incoming/$f ] && cp $X/gold_<host>_incoming/$f $X/gold/$f; done   # judge codes only; never overwrite the local audit/label files


echo "harvest done $(date)"
