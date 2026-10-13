#!/bin/bash
# LOCAL ONLY (no remote access): rebuild every canonical block summary from the raw run directories already under exp/results/{runs,slurm1,slurm2,slurm3,<host>}.
# usage: bash exp/analysis/merge_results.sh     (CPU only; the symlink tree lives on the local disk because exFAT has no symlinks)
X=<path> T=/tmp/p2_merge; GROUPS_="E1R1_core E1R1_sfx E1s_s2 E1s E1s_nt E1s_b0 E1s_math E3_gptoss E3_check a1ctx_main E1_main E1_seeds2 E3_gptoss_s2 E6 E8 E2_lp E2_think E2_cot E7 A6_ix E9_ruis E9_hu_lp E4_load E1_scalar_full E1b E1q E7_full E1_probe E1_ds E1_dsx E1_xl E1_sfx E1_gptoss_ds E1_gptoss_sf E1_pangu E1b_s2 E1q_s2 E10_seeds E9_grice E1p_pop E1_pci_big E1_extra E1p_pop"
rm -rf $T; mkdir -p $T; for g in $GROUPS_; do mkdir -p $T/$g; done
for src in runs slurm1 slurm2 slurm3 slurm4 <host>; do
  for g in $GROUPS_; do
    [ -d $X/results/$src/$g ] || continue
    for d in $X/results/$src/$g/*; do [ -f $d/results.json ] || [ -f $d/math500.json ] || continue; [ -e $T/$g/$(basename $d) ] && continue; ln -s $d $T/$g/$(basename $d); done
  done
done
python3 $X/analysis/concat_seeds.py $T 2>&1 | tail -3
run(){ mode=$1; dst=$2; [ -d $T/$3 ] && [ -n "$(ls $T/$3)" ] || { echo "skip $mode (no $3 runs locally)"; return; }; python3 $X/analysis/export_e1.py $mode $T $X/results/$dst 2>&1 | tail -1; }
run e1s E1S E1s; run e3 E3 E3_gptoss; run a1ctx A1CTX a1ctx_main; run e1 E1 E1_main; run e6 E6 E6; run e8 E8 E8; run e2 E2 E2_lp; run e2t E2T E2_think; run e2c E2C E2_cot
run e1r1 E1R1 E1R1_core; run e1x E1X E1_seeds2; run e3x E3X E3_gptoss_s2; run e7 E7 E7; for g in E7 E7_full; do for src in runs <host> slurm4; do [ -d $X/results/$src/$g ] && python3 $X/analysis/e7_traj.py $X/results/$src/$g | tail -1 > /dev/null; done; done; python3 $X/analysis/e7_marker_merge.py; run a6 A6 A6_ix; run e9 E9 E9_ruis; run e9h E9H E9_hu_lp; run e4 E4 E4_load; run e1f E1SF E1_scalar_full; run e1pci E1PCI E1_pci_big; run e1xt E1XT E1_extra; run e1p E1P E1p_pop; run e1b E1B E1b; run e1q E1Q E1q; run e9g E9G E9_grice; run e7f E7F E7_full; python3 $X/analysis/e7_marker_merge.py; run e1probe E1PROBE E1_probe; run e1ds E1DS E1_ds; run e1dsx E1DSX E1_dsx; run e1xl E1XL E1_xl; run e1sfx E1SFX E1_sfx; run e3ds E1GDS E1_gptoss_ds; run e3sf E1GSF E1_gptoss_sf; run e1pangu E1PANGU E1_pangu; run e10 E10 E10_seeds
mkdir -p $X/results/_superseded; for d in E1S_slurm E3_slurm E1S_b0; do [ -d $X/results/$d ] && mv $X/results/$d $X/results/_superseded/$d.$(date +%H%M%S); done
for f in E1S E3 A1CTX E1 E6 E8 E2 E2T E2C E1X E3X; do [ -f $X/results/$f/summary.json ] && python3 -c "import json;d=json.load(open('$X/results/$f/summary.json'));print('$f', sorted(d)[:12], len(d),'models/ckpts')"; done
python3 $X/analysis/e7_marker_merge.py     # after every export, so E7 and E7F both receive marker/traj blocks

python3 $X/analysis/stats_prereg.py > /tmp/stats_prereg.log 2>&1     # pre-registered statistics (GLMM-style interaction, TOST, Holm) -> results/STATS
