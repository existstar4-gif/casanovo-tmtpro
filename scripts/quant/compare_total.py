# -*- coding: utf-8 -*-
"""compare_total.py — global proteome(Total) 스터디를 PD(Sequest HT · CHIMERYS)와 같은 잣대로 센다 (2026-11-17).

인산화 스터디의 compare_phospho_split.py 와 같은 규칙을 쓰되 인산화로 가르지 않는다.
  · 우리 = N말단 미표지 판독 제외(--labelled-only 와 같음) → bare 서열(I=L)마다 최고 점수 →
          decoy 매칭 FDR 1% 문턱 → 참조 공간(target) 안 서열만
  · PD   = PSMs.txt (Confidence High = PSM 수준 1%) 의 bare 서열. Total 은 PeptideGroups export 가
          스터디마다 있지 않아 PSM 에서 모은다. Study 3 CHIMERYS 는 PeptideGroups 가 있어 차이를 함께 찍는다
  · 단백질 = compare_phospho_split.py 와 같다 (PD: Master Protein Accessions, 우리: 유일 귀속 단백질)
겹침은 bare 서열 기준. PD 서열 가운데 우리 참조 공간 밖의 것은 따로 센다(PD DB 는 TrEMBL 포함).

    python scripts/quant/compare_total.py --psm "out/quant/study3t/*_psm.tsv" --fasta <fasta> \
        --pd-base "data/pd/study3_total_fr_" --label "Study 3 global proteome"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_phospho_split as cps                    # noqa: E402  (read_ours, read_pd, prot_of)
from count_matched_fdr import target_decoy, cutoff_for_fdr   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psm", nargs="+", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--pd-base", required=True, help="'_PSMs.txt' / '-(1)_PSMs.txt' 앞부분")
    ap.add_argument("--pd-pep-check", default=None, help="PeptideGroups 가 있으면 PSM 기준 개수와 비교")
    ap.add_argument("--label", default="study")
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--json")
    a = ap.parse_args()

    paths = []
    for p in a.psm:
        paths.extend(sorted(glob.glob(p)) or [p])
    best, _ph, _bp = cps.read_ours(paths, labelled_only=True)
    tgt, dec = target_decoy(a.fasta, set(best))
    cut, _ = cutoff_for_fdr(best, tgt, dec, a.fdr)
    sel = {s for s, v in best.items() if v >= cut}
    ours = {s for s in sel if s in tgt}
    nd = sum(1 for s in sel if s in dec)
    ours_prot = {tgt[s][0] for s in ours if len(tgt.get(s, ())) == 1}

    pd = {}
    for name, suf in (("Sequest HT", "_PSMs.txt"), ("CHIMERYS", "-(1)_PSMs.txt")):
        ph, no, acc = cps.read_pd(a.pd_base + suf)
        seqs = ph | no
        pd[name] = (seqs, acc)
    ptgt, _ = target_decoy(a.fasta, set().union(*(v[0] for v in pd.values())))

    print("=" * 84)
    print(f"{a.label}  —  global proteome, N-terminally unlabelled reads excluded")
    print(f"  fractions {len(paths)}, our unique sequences {len(best):,}")
    print(f"  decoy-matched FDR {100 * a.fdr:.0f}% threshold = {cut:.4f} (decoy {nd} / target {len(ours)}, "
          f"FDR {100 * nd / max(len(ours), 1):.2f}%)")
    print()
    print("%-14s %12s %12s %10s" % ("method", "peptides", "in our ref.", "proteins"))
    res = {"threshold": cut, "n_decoy": nd}
    for name, (seqs, acc) in pd.items():
        inref = {s for s in seqs if s in ptgt}
        print("%-14s %12s %12s %10s" % (name, f"{len(seqs):,}", f"{len(inref):,}", f"{len(cps.prot_of(seqs, acc)):,}"))
        res[name] = dict(peptides=len(seqs), in_reference=len(inref), proteins=len(cps.prot_of(seqs, acc)))
    print("%-14s %12s %12s %10s" % ("Casanovo-TMT", f"{len(ours):,}", f"{len(ours):,}", f"{len(ours_prot):,}"))
    res["Casanovo-TMT"] = dict(peptides=len(ours), proteins=len(ours_prot))

    s_seq, s_chi = pd["Sequest HT"][0], pd["CHIMERYS"][0]
    either = s_seq | s_chi
    print()
    print("── overlap (bare sequences) ──")
    for name, other in (("Sequest HT", s_seq), ("CHIMERYS", s_chi), ("either PD engine", either)):
        both = ours & other
        print(f"  vs {name:<17} shared {len(both):>8,}  ours only {len(ours - other):>8,}  "
              f"{name} only {len(other - ours):>8,} (in our ref. {len({s for s in other - ours if s in ptgt}):,})")
        res["overlap_" + name] = dict(shared=len(both), ours_only=len(ours - other), pd_only=len(other - ours),
                                       pd_only_in_reference=len({s for s in other - ours if s in ptgt}))
    print(f"  uniquely identified by Casanovo-TMT (absent from both PD engines): {len(ours - either):,} "
          f"({100 * len(ours - either) / max(len(ours), 1):.1f}% of ours)")

    if a.pd_pep_check and os.path.exists(a.pd_pep_check):
        ph, no, _ = cps.read_pd(a.pd_pep_check)
        print(f"\n  check: CHIMERYS PeptideGroups {len(ph | no):,} vs from PSMs {len(s_chi):,}")
        res["check_peptidegroups"] = len(ph | no)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
