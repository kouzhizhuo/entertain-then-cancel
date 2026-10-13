# SIT v2 TEST matrix (METHOD_DESIGN AMENDMENT v2). NOT in any scanned task dir: copy to tasks.d/ ONLY after the dev decision "SIT-check qualifies". Locked roster; NAT, task-aware, SIT-check, neutral check, NT, NAT-512 per model; all arms branch from the same stage-1 thoughts; GSM8K arms via run_gsm.py. Test numbers stay blind until the lead unblocks.
Q=$R/models/shared; B=$R/models/B; LM=(--late_max 256)
GB=(think_full think_b512_plain think_full+taskaware late_sitcheck late_neutral)
gsm(){ nm=$1; mdl=$2; first=$3; shift 3; [ -f $P/M1_test_v2_gsm/$nm/results.json ] || { log "run gsm test v2 $nm"; (cd $EXP/harness && python run_gsm.py --model $mdl --out $P/M1_test_v2_gsm/$nm --split test --late_max 256 --conds $first "${GB[@]}" "$@" > $P/M1_test_v2_gsm_$nm.log 2>&1); log "end gsm test v2 $nm rc=$?"; }; }
vrun M1_test_v2 Qwen3-8B M1_test_v2 Qwen3-8B "${LM[@]}";                                   gsm Qwen3-8B $R/models/Qwen3-8B off
vrun M1_test_v2 $R/models/Qwen3-32B-FP8 M1_test_v2 Qwen3-32B-FP8 "${LM[@]}";                gsm Qwen3-32B-FP8 $R/models/Qwen3-32B-FP8 off
vrun M1_test_v2 $Q/Qwen3.5-9B M1_test_v2 Qwen3.5-9B --max_num_seqs 192 "${LM[@]}";          gsm Qwen3.5-9B $Q/Qwen3.5-9B off --max_num_seqs 192
vrun M1_test_v2 $Q/Qwen3.8-27B M1_test_v2 Qwen3.8-27B --max_num_seqs 192 "${LM[@]}";        gsm Qwen3.8-27B $Q/Qwen3.8-27B off --max_num_seqs 192
PA=(--trust_remote_code --system_file $EXP/harness/pangu_system_prompt.txt --system_always --keep_special --stop_token_ids 45892 --user_suffix_off " /no_think" --off_prefix "[unused16][unused17]" --think_open "[unused16]" --think_close "[unused17]" --think_post "")
vrun M1_test_v2 $B/openPangu-Embedded-7B-V1.1 M1_test_v2 openPangu-Embedded-7B-V1.1 "${PA[@]}" "${LM[@]}"; gsm openPangu-Embedded-7B-V1.1 $B/openPangu-Embedded-7B-V1.1 off "${PA[@]}"
vrun M1_test_v2_it $R/models/DeepSeek-R1-0528-Qwen3-8B M1_test_v2 DeepSeek-R1-0528-Qwen3-8B "${LM[@]}"; gsm DeepSeek-R1-0528-Qwen3-8B $R/models/DeepSeek-R1-0528-Qwen3-8B think_b0_plain
vrun M1_test_v2_it $B/phi4-reasoning M1_test_v2 Phi-4-reasoning "${LM[@]}";                 gsm Phi-4-reasoning $B/phi4-reasoning think_b0_plain
