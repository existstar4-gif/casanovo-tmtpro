#!/usr/bin/env bash
# rerun_phospho_fdr_labelled.sh — rerun_phospho_fdr.sh 와 같되 N말단 미표지 판독을 빼고 센다 (2026-10-30, 사용자 결정).
# PD·FragPipe 는 N말단 TMTpro 를 고정 수식으로 걸어 미표지 펩타이드를 원리상 찾을 수 없다. 단백질 N말단 아세틸은 남긴다.
# 정량 가능 수는 원래 TMT 표지를 요구하므로, 바뀌는 것은 미표지가 빠진 인산화 문턱뿐이다.
set -u
cd "$(dirname "$0")/../.."
export PYTHONUTF8=1
PY=../casanovo_studio/.venv/Scripts/python.exe
FA=out/fasta/mouse_sp_plus_trembl_missing_20260708.fasta
P=data/pd_heldout
run(){ # tag base label
  local tag="$1" b="$2" lab="$3"
  $PY scripts/quant/compare_phospho_split.py --psm out/quant/$tag/*_psm.tsv --fasta "$FA" --fdr-group phospho --labelled-only \
     --pd-pep-a "$P/${b}_PeptideGroups.txt" --pd-pep-b "$P/${b}-(1)_PeptideGroups.txt" --label "$lab (phospho FDR, labelled only)" > out/quant/split_phospho_pfdr_lab_$tag.log 2>&1
  cut=$(grep -o "인산화 문턱 = [0-9.]*" out/quant/split_phospho_pfdr_lab_$tag.log | head -1 | grep -o "[0-9.]*$")
  echo "$tag phospho cut=$cut"
  $PY scripts/quant/quantifiable_share_rule.py --study-dir out/quant/$tag --pd-base "$P/$b" --fasta "$FA" --cut "$cut" \
     --label "$lab (phospho FDR, labelled only)" --json out/quant/quant_share_rule_pfdr_lab_$tag.json > out/quant/quant_share_rule_pfdr_lab_$tag.log 2>&1
}
run study1 "study1_phospho_fr_0" "Study 1"
run study2 "study2_phospho_fr_"  "Study 2"
run study3  "study3_phospho_fr_"    "Study 3"
echo PFDR_LAB_DONE
