#!/usr/bin/env bash
set -euo pipefail

hosts=(
  cip7b1 cip7b2 cip7c0 cip7c1 cip7c2 cip7d0 cip7d1 cip7d2 cip7e0
  cip7e1 cip7e2 cip7f0 cip7f1 cip7f2 cip7g0 cip7g1 cip7g2 cip3a0
)

parquet_files=(
  case24_ieee_rts_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
  case30_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
  case33bw_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
  case39_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
  case57_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
  case89pegase_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
  case118_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
  case145_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
  case300_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet
)

osascript -e 'tell application "Terminal" to activate' \
          -e 'tell application "Terminal" to if (count of windows) is 0 then do script ""'

for i in $(seq 1 18); do
  host="${hosts[$((i-1))]}"
  if [ "$i" -le 9 ]; then
    f="${parquet_files[$((i-1))]}"
    armijo='--use_armijo'
  else
    j=$((i-10))
    f="${parquet_files[$j]}"
    armijo=''
  fi

  if [ "$i" -gt 1 ]; then
    osascript -e 'tell application "System Events" to tell process "Terminal" to keystroke "t" using command down' \
              -e 'delay 0.2'
  fi

  train_cmd="python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/$f --vlimit $armijo --model GNSMsg_EdgeSelfAttn"

  osascript -e "tell application \"Terminal\" to do script \"ssh $host\" in selected tab of front window" \
            -e 'delay 1.0' \
            -e 'tell application "Terminal" to do script "clear" in selected tab of front window' \
            -e 'tell application "Terminal" to do script "conda activate /proj/aimi-adl/envs/adl23_2/" in selected tab of front window' \
            -e 'tell application "Terminal" to do script "export PYTHONPATH=~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC:$PYTHONPATH" in selected tab of front window' \
            -e 'tell application "Terminal" to do script "cd ~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC" in selected tab of front window' \
            -e "tell application \"Terminal\" to do script \"$train_cmd\" in selected tab of front window"
done
