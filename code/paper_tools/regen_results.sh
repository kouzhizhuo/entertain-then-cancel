#!/bin/bash
# Regenerate numbers/results.tex from every merged summary block (see notes/RESULTS_KEYS.md), then build.
set -e
cd "$(dirname "$0")/.."
R=../exp/results; G=../exp/gold
python3 ../exp/analysis/m1dev_summary.py >/dev/null
python3 tools/json2tex.py \
  $R/pilot/summary_pilot.json:pilot $R/E1/summary.json:e1 $R/E1S/summary.json:e1s $R/A1CTX/summary.json:a1ctx \
  $R/E3/summary.json:e3 $R/E2/summary.json:e2 $R/E2T/summary.json:e2t $R/E2C/summary.json:e2c $R/E6/summary.json:e6 \
  $R/E8/summary.json:e8 $R/E1F/summary.json:e1f $R/E7/summary.json:e7 $R/E7F/summary.json:e7f $R/E4/summary.json:e4 $R/E9/summary.json:e9 $R/E9G/summary.json:e9g $R/E1P/summary.json:e1p $R/E1B/summary.json:e1b $R/E1SF/summary.json:e1sf $R/E1XT/summary.json:e1xt $R/E1PCI/summary.json:e1pci $R/A6/summary.json:a6 $R/E1PROBE/summary.json:e1probe $R/E1Q/summary.json:e1q $R/E1DS/summary.json:e1ds $R/E1DSX/summary.json:e1dsx $R/E1SFX/summary.json:e1sfx $R/E1GDS/summary.json:e1gds $R/E1PANGU/summary.json:e1pangu $R/DERIVED/summary.json:derived $R/E10/summary.json:e10 $R/E1XL/summary.json:e1xl $R/M1DEV/summary.json:m1dev $R/E1GSF/summary.json:e1gsf $R/E1R1/summary.json:e1r1 $R/M1/x3_test.json:x3test \
  $G/summary_gold.json:gold $G/summary_human_v4_tex.json:gold $G/summary_gpt6_tex.json:gold $G/summary_judge_tex.json:gold \
  ${EXTRA_BLOCKS:-} -o build/results_full.tex --digits 1
python3 tools/make_tab_ablation.py >/dev/null; python3 tools/make_app_v2.py >/dev/null; python3 tools/make_fig_size.py >/dev/null; python3 tools/make_tab_sens.py >/dev/null
python3 tools/filter_results.py build/results_full.tex numbers/results.tex
python3 tools/captions_below.py tables/*.tex sections/*.tex >/dev/null
python3 tools/style_tables.py tables/tab_*.tex sections/A_all.tex sections/A_floats.tex >/dev/null
