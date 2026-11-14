# -*- coding: utf-8 -*-
"""compare_phospho_split.py — 인산화 / 비인산화를 **갈라서** 같은 FDR 로 센다 (2026-09-21).

왜 필요한가 (사용자 지적):
인산화 스터디의 개수 비교가 사실은 인산화끼리의 비교가 아니었다. 실측하면

  PD 동정  : 인산화 98.4%(Sequest) / 97.6%(CHIMERYS)  — 농축이 잘 됐다
  우리 예측 : 인산화 **71.2%**                         — 29%가 비인산화다

즉 우리 쪽 개수에는 비인산화가 PD 보다 훨씬 많이 섞여 있다. 그대로 맞대면
'인산화를 얼마나 찾나'를 재는 게 아니다. 그래서 양쪽을 같은 규칙으로 가른다.

규칙(양쪽 동일): 한 bare 서열이 **인산화 형태로 한 번이라도 관측되면** 인산화로 센다.
  · 우리  = `peptidoform` 열(2번)에 `Phospho` 가 든 PSM 이 하나라도 있으면
  · PD    = `PeptideGroups` 의 `Modifications` 에 `Phospho` 가 든 행이 하나라도 있으면

⚠️ FDR 문턱은 **전체에서 한 번** 잡고 각 부류에 적용한다. 부류마다 따로 잡으면
   운영점이 달라져 또 다른 것을 비교하게 된다.
⚠️ 단백질은 PD `Proteins.txt` 로는 가를 수 없어(수식 정보 없음)
   `PeptideGroups` 의 `Master Protein Accessions` 로 센다. 그래서 PD 의 단백질 수가
   `count_matched_fdr.py` 가 찍는 값과 조금 다르다 — **같은 규칙끼리만 비교할 것**.

사용:
  python scripts/quant/compare_phospho_split.py --psm "out/quant/htb/*_psm.tsv" \
      --fasta <fasta> --pd-pep-a <Sequest _PeptideGroups.txt> \
      --pd-pep-b <CHIMERYS _PeptideGroups.txt> --label "<스터디>"
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "casanovo_studio"))
sys.path.insert(0, os.path.join(BASE, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.fasta import normalize                      # noqa: E402
import parse_psms as pp                               # noqa: E402
from count_matched_fdr import target_decoy, cutoff_for_fdr   # noqa: E402

MOD = re.compile(r"\[[^\]]*\]")


def bare(s):
    return MOD.sub("", s or "").replace("-", "").replace(".", "").strip().upper()


def read_ours(paths, labelled_only=False):
    """{bare 서열: 최고 점수}, 인산화 형태로 본 적 있는 서열 집합, {bare 서열: 인산화 판독 중 최고 점수}.
    labelled_only: N말단에 수식이 없는(TMTpro 도 아세틸도 아닌) 판독은 뺀다 — PD·FragPipe 는 N말단 TMTpro 고정이라 찾을 수 없다 (2026-10-30)."""
    best, phos, best_p = {}, set(), {}
    for p in paths:
        seen = False
        for line in io.open(p, encoding="utf-8"):
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[:2] == ["stem", "scan"]:
                seen = True
                continue
            if not seen:
                continue
            try:
                sc = float(c[4])
            except (ValueError, IndexError):
                continue
            pf = c[2]
            if labelled_only and not pf.startswith("["):
                continue
            k = bare(pf)
            if not k:
                continue
            if "phospho" in pf.lower():
                phos.add(k)
                if k not in best_p or sc > best_p[k]:
                    best_p[k] = sc
            if k not in best or sc > best[k]:
                best[k] = sc
    return best, phos, best_p


def read_pd(path):
    """PD PeptideGroups -> (인산화 서열집합, 비인산화 서열집합, {서열: master accessions})."""
    ph, no, acc = set(), set(), collections.defaultdict(set)
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        hdr = [pp.clean_field(c) for c in fh.readline().rstrip("\n").split("\t")]
        ix = {x: i for i, x in enumerate(hdr)}
        cs, cm, ca = ix.get("Sequence"), ix.get("Modifications"), ix.get("Master Protein Accessions")
        # 일부 CHIMERYS export(2026-08 이후)는 Sequence 열이 없고 Annotated Sequence 만 있다
        ann = cs is None
        if ann:
            cs = ix.get("Annotated Sequence")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if cs is None or cs >= len(c):
                continue
            s = pp.clean_field(c[cs])
            s = (pp.extract_peptide(s) or "") if ann else s
            if not s:
                continue
            k = normalize(s.upper())
            mods = pp.clean_field(c[cm]) if (cm is not None and cm < len(c)) else ""
            (ph if "phospho" in mods.lower() else no).add(k)
            if ca is not None and ca < len(c):
                for a in pp.clean_field(c[ca]).split(";"):
                    a = a.strip()
                    if a:
                        acc[k].add(a)
    no -= ph                      # 인산화 형태로도 본 서열은 인산화로 센다
    return ph, no, acc


def prot_of(seqs, acc):
    out = set()
    for s in seqs:
        out |= acc.get(s, set())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psm", nargs="+", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--pd-pep-a", required=True)
    ap.add_argument("--pd-pep-b", default=None)
    ap.add_argument("--label", default="study")
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--fdr-group", choices=["all", "phospho"], default="all",
                    help="phospho = 인산화 판독만으로 FDR 문턱을 따로 잡는다 (2026-10-29, 우리 분석법으로 채택). "
                         "인산화 서열은 인산화 판독의 최고 점수로, 비인산화 서열은 비인산화끼리 잡은 문턱으로 거른다")
    ap.add_argument("--labelled-only", action="store_true",
                    help="N말단 미표지 판독 제외 (2026-10-30, 사용자 결정: 비교 도구가 원리상 찾을 수 없는 판독)")
    a = ap.parse_args()

    paths = []
    for p in a.psm:
        paths.extend(sorted(glob.glob(p)) or [p])
    best, ours_phos, best_p = read_ours(paths, a.labelled_only)
    print("=" * 78)
    print(a.label + "  —  인산화 / 비인산화 분리")
    print("  분획 %d개, 우리 고유 서열 %s개" % (len(paths), f"{len(best):,}"))

    tgt, dec = target_decoy(a.fasta, set(best))
    cut, _ = cutoff_for_fdr(best, tgt, dec, a.fdr)
    if cut is None:
        print("  ⚠️ FDR 문턱 없음 — 거르지 않고 센다")
        sel = set(best)
    else:
        print("  ★ decoy 매칭 FDR %.0f%% 문턱 = %.4f (전체에서 한 번 잡아 양쪽에 적용)"
              % (100 * a.fdr, cut))
        sel = {s for s, sc in best.items() if sc >= cut}
    ours_in = {s for s in sel if s in tgt}
    ours_p = {s for s in ours_in if s in ours_phos}
    ours_n = ours_in - ours_p
    if a.fdr_group == "phospho":
        cut_p, _ = cutoff_for_fdr(best_p, tgt, dec, a.fdr)
        bn = {s: v for s, v in best.items() if s not in ours_phos}
        cut_n, _ = cutoff_for_fdr(bn, tgt, dec, a.fdr)
        ours_p = {s for s, v in best_p.items() if v >= cut_p and s in tgt}
        ours_n = {s for s, v in bn.items() if v >= cut_n and s in tgt}
        dp = sum(1 for s, v in best_p.items() if v >= cut_p and s in dec)
        dn = sum(1 for s, v in bn.items() if v >= cut_n and s in dec)
        print("  ★ 그룹별 FDR: 인산화 문턱 = %.4f (decoy %d / target %d), 비인산화 문턱 = %.4f (decoy %d / target %d), 합산 FDR %.2f%%"
              % (cut_p, dp, len(ours_p), cut_n, dn, len(ours_n), 100.0 * (dp + dn) / max(len(ours_p) + len(ours_n), 1)))

    def ours_prot(seqs):
        return {tgt[s][0] for s in seqs if len(tgt.get(s, ())) == 1}

    rows = []
    pa = read_pd(a.pd_pep_a)
    rows.append(("Sequest HT (PD)", pa))
    if a.pd_pep_b and os.path.exists(a.pd_pep_b):
        rows.append(("CHIMERYS (PD)", read_pd(a.pd_pep_b)))

    print()
    print("── 펩타이드 (bare 서열) ─────────────────────────────")
    print("%-22s %12s %12s %12s" % ("방법", "인산화", "비인산화", "인산화 비율"))
    for name, (ph, no, _acc) in rows:
        tot = len(ph) + len(no)
        print("%-22s %12s %12s %11.1f%%" % (name, f"{len(ph):,}", f"{len(no):,}",
                                            100.0 * len(ph) / max(tot, 1)))
    tot = len(ours_p) + len(ours_n)
    print("%-22s %12s %12s %11.1f%%" % ("우리 모델", f"{len(ours_p):,}", f"{len(ours_n):,}",
                                        100.0 * len(ours_p) / max(tot, 1)))

    print()
    print("── 단백질 (인산화 펩타이드가 받치는 것) ─────────────")
    print("%-22s %12s %12s" % ("방법", "인산화", "비인산화"))
    for name, (ph, no, acc) in rows:
        print("%-22s %12s %12s" % (name, f"{len(prot_of(ph, acc)):,}",
                                   f"{len(prot_of(no, acc)):,}"))
    print("%-22s %12s %12s" % ("우리 모델", f"{len(ours_prot(ours_p)):,}",
                               f"{len(ours_prot(ours_n)):,}"))
    print()
    print("  ⚠️ PD 단백질은 PeptideGroups 의 Master Protein Accessions 로 셌다")
    print("     (Proteins.txt 는 수식 정보가 없어 가를 수 없다). 우리는 유일 펩타이드만.")
    print("  ⚠️ 우리 FDR 은 '서열이 참조 공간에 우연히 있을 확률'이고 엔진 FDR 과 다른 양이다.")


if __name__ == "__main__":
    main()
