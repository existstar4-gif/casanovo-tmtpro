# -*- coding: utf-8 -*-
"""overlap4_phospho.py — 미학습 인산화 Study 1–3 에서 네 방법의 인산화 펩타이드 겹침 (2026-11-12).

Fig. 4A 와 같은 규칙으로 네 집합을 만든다:
  SEQUEST HT, CHIMERYS  = PD PeptideGroups 의 인산화 bare 서열 (compare_phospho_split.read_pd)
  FragPipe              = 자기 기본 출력 psm.tsv 의 인산화 bare 서열 (A_default, 2026-11-13)
  Casanovo-TMT          = N말단 미표지 제외 + 인산화 그룹 1% FDR + target 서열 (compare_phospho_split --fdr-group phospho --labelled-only)
집합 크기가 Fig. 4A 막대와 같아야 한다 — 다르면 멈춘다.

    python scripts/quant/overlap4_phospho.py
출력: out/quant/validation/overlap4_<study>.json  (교집합 15칸, UpSet 용)
"""
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_phospho_split as cps          # noqa: E402
import fragpipe_phospho_compare as fpc       # noqa: E402
from count_matched_fdr import cutoff_for_fdr, target_decoy  # noqa: E402
from rescued_pick import FA, P, BASES        # noqa: E402

FP_ST = {"study3": "s3", "study2": "s2", "study1": "s1"}
METHODS = ["SEQUEST HT", "CHIMERYS", "FragPipe", "Casanovo-TMT"]


def ours(study):
    paths = sorted(glob.glob(f"out/quant/{study}/*_psm.tsv"))
    best, ours_phos, best_p = cps.read_ours(paths, labelled_only=True)
    tgt, dec = target_decoy(FA, set(best))
    cut_p, _ = cutoff_for_fdr(best_p, tgt, dec, 0.01)
    return {s for s, v in best_p.items() if v >= cut_p and s in tgt}


def fragpipe(study):
    """2026-11-13: FragPipe 자기 기본 출력(Philosopher psm.tsv, 1%)의 인산화 bare 서열 — Fig. 4A 와 같다."""
    return fpc.phospho_bare(fpc.read_default(FP_ST[study]))[0]


def main():
    for study in ("study1", "study2", "study3"):
        sets = {
            "SEQUEST HT": cps.read_pd(P + BASES[study] + "_PeptideGroups.txt")[0],
            "CHIMERYS": cps.read_pd(P + BASES[study] + "-(1)_PeptideGroups.txt")[0],
            "FragPipe": fragpipe(study),
            "Casanovo-TMT": ours(study),
        }
        sizes = {m: len(s) for m, s in sets.items()}
        # 정확히 한 조합에만 속하는 수 (UpSet)
        exact = {}
        allp = set().union(*sets.values())
        for r in range(1, 5):
            for combo in itertools.combinations(METHODS, r):
                inn = set.intersection(*(sets[m] for m in combo))
                out = set().union(*(sets[m] for m in METHODS if m not in combo)) if r < 4 else set()
                exact[" & ".join(combo)] = len(inn - out)
        assert sum(exact.values()) == len(allp)
        res = dict(study=study, sizes=sizes, union=len(allp), exact=exact)
        json.dump(res, open(f"out/quant/validation/overlap4_{study}.json", "w", encoding="utf-8"), indent=2)
        print(study, sizes, "union", len(allp), flush=True)


if __name__ == "__main__":
    main()
