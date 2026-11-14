# -*- coding: utf-8 -*-
"""export_validation_groups.py — 검증 집단별 대표 PSM 을 표로 내보낸다 (2026-11-03, 예측 스펙트럼 유사도용).

집단 정의는 ours_only_validation.py 와 같다. 오답 기준선은 셋으로 나눈다:
  wrong_decoy (역순 서열), wrong_arath (애기장대 entrapment), wrong_swap (이웃 잔기 맞바꾼 생쥐 서열 — 순서만 틀린 오판독)
펩타이드마다 최고 점수 인산화 PSM 하나. 오답은 점수 전 범위.

    python scripts/quant/export_validation_groups.py --study study3
출력: out/quant/validation/groups_<study>.tsv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import count_matched_fdr as cm                # noqa: E402
import compare_phospho_split as cps           # noqa: E402
import fragpipe_phospho_compare as fpc        # noqa: E402
import ours_only_validation as ov             # noqa: E402
from entrapment_formal import swap_entrapment  # noqa: E402
from rescued_pick import FA, P, BASES         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="study3")
    a = ap.parse_args()
    psms = ov.read_psms(a.study)
    best, best_p = {}, {}
    for p in psms:
        k = p["bare"]
        if p["score"] > best.get(k, (-1,))[0]:
            best[k] = (p["score"], p)
        if "Phospho" in p["pf"] and p["score"] > best_p.get(k, (-1,))[0]:
            best_p[k] = (p["score"], p)
    tgt, dec = cm.target_decoy(FA, set(best))
    joint, _ = cm.cutoff_for_fdr({k: v[0] for k, v in best.items()}, tgt, dec, 0.01)
    cut_p, _ = cm.cutoff_for_fdr({k: v[0] for k, v in best_p.items()}, tgt, dec, 0.01)
    mouse = ov.proteins(FA)
    t_all = ov.full_digest(mouse.values()); d_all = ov.full_digest(mouse.values(), rev=True)
    e_all = ov.full_digest(ov.proteins(ov.ENT).values()) - t_all - d_all
    sw_all = set(swap_entrapment(t_all, t_all | d_all).values())
    del t_all, d_all
    other = ov.engine_phospho(a.study)   # 2026-11-13: Fig. 4C 기준 (엔진의 인산화 동정)
    rows = []
    for k, (s, p) in best_p.items():
        g = None
        if k in tgt and s >= cut_p:
            g = "shared" if k in other else "ours_conf"   # 2026-11-13: 인산화 그룹 기준 하나로 통일
        elif k in dec:
            g = "wrong_decoy"
        elif k in e_all:
            g = "wrong_arath"
        elif k in sw_all:
            g = "wrong_swap"
        if g:
            rows.append(dict(group=g, bare=k, pf=p["pf"], stem=p["stem"], scan=p["scan"], charge=p["z"], score=round(s, 4),
                             above_cut=int(s >= cut_p)))
    os.makedirs("out/quant/validation", exist_ok=True)
    out = f"out/quant/validation/groups_{a.study}.tsv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
    import collections
    print(a.study, dict(collections.Counter(r["group"] for r in rows)), "->", out)


if __name__ == "__main__":
    main()
