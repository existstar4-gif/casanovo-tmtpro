#!/usr/bin/env bash
# run_study_generic.sh <출력하위폴더> <트립신인덱스> <stem목록파일>
# 한 스터디의 모든 분획에 정량 파이프라인을 돌린다. 이미 있는 산출물은 건너뛴다(재개 가능).
# 환경변수 PLEX=18 이면 18-plex 리포터를 뽑는다(기본 16). Studies 1 and 2 가 18-plex 다(09-28).
set -u
cd "$(dirname "$0")/../.."
SUB="$1"; IDX="$2"; LIST="$3"
EXE=./.venv-casanovo/Scripts/casanovo.exe
export PYTHONUTF8=1
CKPT='models/casanovo_tmt.ckpt'
MZDIR='data/mzml_tmt'
OUT="out/quant/$SUB"
mkdir -p "$OUT"
while read -r stem; do
  [ -z "$stem" ] && continue
  echo "########## $stem  $(date '+%H:%M:%S')"
  MGF="$OUT/$stem.mgf"; MZT="$OUT/$stem.mztab"; REP="$OUT/$stem.rep.tsv"
  # 변환이 실패하면 반쪽 MGF 를 지운다 — 남겨 두면 재실행 때 [ -f ] 가 통과해 반쪽을 조용히 시퀀싱한다 (09-28)
  [ -f "$MGF" ] || python scripts/quant/mzml_to_mgf.py "$MZDIR/$stem.mzML" -o "$MGF" || { rm -f "$MGF"; continue; }
  [ -f "$REP" ] || python scripts/quant/extract_reporters.py "$MZDIR/$stem.mzML" -o "$REP" \
      --plex "${PLEX:-16}" --tol 0.003 2>&1 | tail -1
  if [ ! -f "$MZT" ]; then
    unset CASANOVO_GATE CASANOVO_GATE_RESCUE CASANOVO_TMT_MARGIN CASANOVO_TMT_LAMBDA CASANOVO_KNAPSACK_BONUS
    export CASANOVO_KNAPSACK=1 CASANOVO_DBRANK=1 CASANOVO_DBRANK_INDEX="$IDX" CASANOVO_DBRANK_BONUS=1.00
    $EXE sequence -c out/eval_final.yaml -m "$CKPT" -d "$OUT" -o "$stem" -f "$MGF" 2>&1 \
      | grep -Ei 'Sequenced|rror' | head -2
  fi
  if [ -f "$MZT" ] && [ -f "$REP" ]; then
    [ -f "$OUT/${stem}_psm.tsv" ] || python scripts/quant/combine_psm_quant.py \
      --mztab "$MZT" --mgf "$MGF" --reporters "$REP" --out-prefix "$OUT/$stem" --index "$IDX" \
      2>&1 | tail -2
    # 디스크를 아끼려고 중간 MGF 는 지운다(원본 mzML 로 언제든 재생성된다)
    rm -f "$MGF"
  fi
done < "$LIST"
echo "STUDY_DONE $SUB $(date '+%H:%M:%S')"
