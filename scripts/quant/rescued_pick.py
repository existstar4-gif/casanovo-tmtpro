# -*- coding: utf-8 -*-
"""rescued_pick.py — 인산화 그룹 FDR 로 새로 통과한(공동 문턱에선 떨어지던) 인산화 펩타이드 목록과 수동 확인용 상위 10개 (2026-10-29, 사용자 요청).

구제 = 인산화 판독 최고 점수가 인산화 문턱 이상 · 공동 문턱 미만 · target(참조 공간 안).
펩타이드마다 최고 점수 스캔 하나. '가장 좋은 것' = 점수 순, 전구체 질량 일치(|ppm| ≤ 20) 인 것만.
절단 자리 커버리지(chimeric_case_study.frag_cover)와 PD(Sequest / CHIMERYS) 동정 여부를 함께 적는다.

    python scripts/quant/rescued_pick.py --study study3
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from pyteomics import mzml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import count_matched_fdr as cm                # noqa: E402
import chimeric_case_study as cs              # noqa: E402
import compare_phospho_split as cps           # noqa: E402

FA = "out/fasta/mouse_sp_plus_trembl_missing_20260708.fasta"
P = "data/pd/"
BASES = {"study3": "study3_phospho_fr_", "study2": "study2_phospho_fr_",
         "study1": "study1_phospho_fr_0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="study3")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(f"out/quant/{a.study}/*_psm.tsv")):
        rd = csv.DictReader((ln for ln in open(p, encoding="utf-8") if not ln.startswith("#")), delimiter="\t")
        for r in rd:
            try:
                r["score"] = float(r["score"]); r["charge"] = int(r["charge"])
            except ValueError:
                continue
            r["bare"] = cps.bare(r["peptidoform"])
            rows.append(r)
    best = {}
    best_p = {}
    for r in rows:
        k = r["bare"]
        if r["score"] > best.get(k, -1):
            best[k] = r["score"]
        if "Phospho" in r["peptidoform"] and r["score"] > best_p.get(k, (-1,))[0]:
            best_p[k] = (r["score"], r)
    tgt, dec = cm.target_decoy(FA, set(best))
    joint, _ = cm.cutoff_for_fdr(best, tgt, dec, 0.01)
    cut_p, _ = cm.cutoff_for_fdr({k: v[0] for k, v in best_p.items()}, tgt, dec, 0.01)
    pd = {}
    for name, suf in (("Sequest", "_PeptideGroups.txt"), ("CHIMERYS", "-(1)_PeptideGroups.txt")):
        ph, no, _ = cps.read_pd(P + BASES[a.study] + suf)
        pd[name] = ph | no
    resc = [(sc, r) for k, (sc, r) in best_p.items() if k in tgt and cut_p <= sc < joint]
    resc.sort(key=lambda x: -x[0])
    out_dir = f"out/quant/rescued_{a.study}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/rescued_all.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["stem", "scan", "peptidoform", "score", "ppm", "charge", "Sequest", "CHIMERYS", "proteins"])
        for sc, r in resc:
            w.writerow([r["stem"], r["scan"], r["peptidoform"], sc, r["ppm"], r["charge"],
                        int(r["bare"] in pd["Sequest"]), int(r["bare"] in pd["CHIMERYS"]), ";".join(tgt[r["bare"]])])
    pick = []
    for sc, r in resc:
        try:
            if abs(float(r["ppm"])) > 20:
                continue
        except ValueError:
            continue
        with mzml.MzML(f"data/mzml_tmt/{r['stem']}.mzML", use_index=True) as rd:
            sp = rd.get_by_id(f"controllerType=0 controllerNumber=1 scan={r['scan']}")
        mz = np.asarray(sp["m/z array"], float); it = np.asarray(sp["intensity array"], float)
        seq, mods = cs.our_form(r["peptidoform"])
        cov = cs.frag_cover(seq, mods, r["charge"], mz, it)
        pick.append(dict(stem=r["stem"], scan=int(r["scan"]), peptidoform=r["peptidoform"], score=sc, ppm=float(r["ppm"]),
                         charge=r["charge"], frag_cover=cov, Sequest=r["bare"] in pd["Sequest"],
                         CHIMERYS=r["bare"] in pd["CHIMERYS"], proteins=tgt[r["bare"]]))
        if len(pick) >= a.n:
            break
    summ = dict(joint_cut=joint, phospho_cut=cut_p, n_rescued=len(resc),
                rescued_in_Sequest=sum(1 for _, r in resc if r["bare"] in pd["Sequest"]),
                rescued_in_CHIMERYS=sum(1 for _, r in resc if r["bare"] in pd["CHIMERYS"]), top=pick)
    json.dump(summ, open(f"{out_dir}/top{a.n}.json", "w", encoding="utf-8"), indent=2, default=str)
    print(json.dumps(summ, indent=2, default=str))


if __name__ == "__main__":
    main()
