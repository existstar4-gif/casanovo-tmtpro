# -*- coding: utf-8 -*-
"""count_matched_fdr.py — **같은 잣대**로 펩타이드·단백질 개수를 센다 (2026-09-18).

왜 필요한가:
지금까지의 개수 비교는 공평하지 않았다. PD(Sequest·CHIMERYS)는 **FDR 1%** 로 걸러진
결과인데 우리는 **아무 기준도 걸지 않은** 전체였다. 그대로 "우리가 더 많이 찾았다"고 하면
잣대가 다른 것을 비교한 것이고 심사에서 바로 무너진다.

그래서 우리 쪽에도 같은 수준의 기준을 세운다 — **decoy 매칭 기반 FDR 1%**.

  · 우리 예측 서열을 참조 프로테옴(target)과 그것을 뒤집은 것(decoy)에 각각 대조한다.
  · 점수를 높은 쪽부터 낮추며 누적 decoy 비율을 본다.
  · 그 비율이 1% 를 넘기 직전의 점수를 문턱으로 삼는다.

⚠️ 이 값을 **검색 엔진 FDR 과 같다고 말하지 않는다.** 엔진 FDR 은 '이 동정이 틀릴 확률'을
   target-decoy 경쟁으로 추정한 것이고, 우리 값은 '이 서열이 참조 공간에 우연히 있을 확률'의
   추정이다. 다른 양이다. 용어를 `estimated FDR (decoy-matched)` 로 고정한다
   (`GUI_PLAN.md` §1 의 원칙과 같다).
   그럼에도 **둘 다 1% 로 맞추면 비교의 공정성은 크게 좋아진다** — 아무것도 안 거는 것보다 낫다.

사용:
  python scripts/quant/count_matched_fdr.py --psm "out/quant/htb/*_psm.tsv" \\
      --fasta <fasta> --pd-a <Sequest _Proteins.txt> --pd-b <CHIMERYS _Proteins.txt> \\
      --pd-pep-a <Sequest _PeptideGroups.txt> --pd-pep-b <CHIMERYS _PeptideGroups.txt> \\
      --label "<스터디>"
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "casanovo_studio"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from core.fasta import read_fasta, digest, normalize              # noqa: E402
from core.quant_rollup import accession_of, normalize  # noqa: E402
import parse_psms as pp        # noqa: E402

MOD = re.compile(r"\[[^\]]*\]")
FIXED = ["stem", "scan", "peptidoform", "sequence", "score", "ppm",
         "charge", "purity", "n_found", "in_db"]


def bare(s):
    return MOD.sub("", s or "").replace("-", "").replace(".", "").strip().upper()


def read_ours(paths):
    """{bare 서열: 최고 점수} — 펩타이드 단위로 가장 좋은 점수를 남긴다."""
    best = {}
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
            k = bare(c[3])
            if k and (k not in best or sc > best[k]):
                best[k] = sc
    return best


def target_decoy(fasta, wanted):
    """관측 서열을 target/decoy 에 각각 대조. {서열: 'target'|'decoy'|'none'}, 단백질 매핑."""
    want = set(wanted)
    tgt, dec = {}, set()
    for header, seq in read_fasta([fasta]):
        if not seq:
            continue
        acc = accession_of(header)
        for pep in digest(seq, 2, 6, 50, met_loss=True):
            k = normalize(pep)
            if k in want:
                tgt.setdefault(k, set()).add(acc)
        for pep in digest(seq[::-1], 2, 6, 50, met_loss=True):
            k = normalize(pep)
            if k in want:
                dec.add(k)
    dec -= set(tgt)                   # 양쪽에 있으면 target 우선
    return {k: sorted(v) for k, v in tgt.items()}, dec


def cutoff_for_fdr(best, tgt, dec, target_fdr=0.01):
    """점수를 높은 쪽부터 낮추며 누적 decoy 비율이 target_fdr 을 넘기 직전 점수."""
    rows = sorted(((sc, s) for s, sc in best.items()), reverse=True)
    nt = nd = 0
    cut, kept = None, 0
    for sc, s in rows:
        if s in tgt:
            nt += 1
        elif s in dec:
            nd += 1
        else:
            continue                  # 참조 공간 밖 — FDR 계산에 넣지 않는다
        if nt == 0:
            continue
        fdr = nd / nt
        if fdr <= target_fdr:
            cut, kept = sc, nt
    return cut, kept


def read_pd_peptides(path):
    """PD PeptideGroups 를 **두 단위로** 센다 — (펩타이도폼 행 수, 고유 bare 서열 수).

    ★ 2026-09-21 (사용자 지적): PD 는 `Sequence` + `Modifications` 를 키로 쓰고 우리는 bare
    서열로 센다. 인산화 실험에서는 같은 서열의 부위 이성체가 각각 한 행이 되므로, 그대로
    맞대면 **단위가 다른 것을 비교**하게 된다. 실측 부풀림은 Total 1.04배 vs Phospho 1.18~1.21배
    로, 하필 우리가 지는 쪽에서 편향이 4~5배 크다. 따라서 둘 다 찍고 **bare 열을 공정한 쌍**으로
    삼는다. I/L 은 양쪽 다 통합한다.
    """
    rows = 0
    seqs = set()
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        hdr = [pp.clean_field(c) for c in fh.readline().rstrip("\n").split("\t")]
        ix = {x: i for i, x in enumerate(hdr)}
        cs = ix.get("Sequence")
        # 일부 CHIMERYS export(2026-08 이후)는 Sequence 열이 없고 Annotated Sequence 만 있다
        ann = cs is None
        if ann:
            cs = ix.get("Annotated Sequence")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            rows += 1
            if cs is not None and cs < len(c):
                s = pp.clean_field(c[cs])
                s = (pp.extract_peptide(s) or "") if ann else s
                if s:
                    seqs.add(normalize(s.upper()))
    return rows, len(seqs)


def read_pd_count(path, key_col_candidates):
    n = 0
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        hdr = [pp.clean_field(c) for c in fh.readline().rstrip("\n").split("\t")]
        ix = {x: i for i, x in enumerate(hdr)}
        im = ix.get("Master")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if im is not None and im < len(c) and "Master" not in pp.clean_field(c[im]):
                continue
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psm", nargs="+", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--pd-a", required=True)
    ap.add_argument("--pd-b", default=None, help="CHIMERYS 분석이 없으면 생략")
    ap.add_argument("--pd-pep-a", default=None)
    ap.add_argument("--pd-pep-b", default=None)
    ap.add_argument("--label", default="study")
    ap.add_argument("--fdr", type=float, default=0.01)
    a = ap.parse_args()

    paths = []
    for p in a.psm:
        paths.extend(sorted(glob.glob(p)) or [p])
    best = read_ours(paths)
    print("=" * 78)
    print(a.label)
    print("  분획 %d개, 우리 고유 서열 %s개 (거름 없음)" % (len(paths), f"{len(best):,}"))

    tgt, dec = target_decoy(a.fasta, set(best))
    n_t = sum(1 for s in best if s in tgt)
    n_d = sum(1 for s in best if s in dec)
    n_o = len(best) - n_t - n_d
    print("  참조 대조 — target %s / decoy %s / 참조 밖 %s"
          % (f"{n_t:,}", f"{n_d:,}", f"{n_o:,}"))

    cut, kept = cutoff_for_fdr(best, tgt, dec, a.fdr)
    print()
    if cut is None:
        print("  ⚠️ FDR %.0f%% 를 만족하는 점수 문턱이 없다 — 거르지 않은 값만 보고한다"
              % (100 * a.fdr))
        sel = set(best)
    else:
        print("  ★ decoy 매칭 FDR %.0f%% 문턱 = 점수 %.4f" % (100 * a.fdr, cut))
        sel = {s for s, sc in best.items() if sc >= cut}
    pep_ours = sum(1 for s in sel if s in tgt)          # 참조 안에 있는 것만 센다
    prot = collections.Counter()
    for s in sel:
        accs = tgt.get(s)
        if accs and len(accs) == 1:                     # 유일 펩타이드만
            prot[accs[0]] += 1
    print()
    print("── 같은 잣대(FDR %.0f%%)로 센 개수 ─────────────────────" % (100 * a.fdr))
    print("%-22s %12s %12s %12s" % ("방법", "bare 서열", "펩타이도폼", "단백질"))
    print("%-22s %12s %12s %12s" % ("", "(공정한 쌍)", "(PD 원래 단위)", ""))
    pa = read_pd_peptides(a.pd_pep_a) if (a.pd_pep_a and os.path.exists(a.pd_pep_a)) else None
    pb = read_pd_peptides(a.pd_pep_b) if (a.pd_pep_b and os.path.exists(a.pd_pep_b)) else None
    fmt = lambda v: f"{v:,}" if v else "-"
    print("%-22s %12s %12s %12s" % ("Sequest HT (PD)", fmt(pa and pa[1]), fmt(pa and pa[0]),
                                    fmt(read_pd_count(a.pd_a, None))))
    if a.pd_b and os.path.exists(a.pd_b):
        print("%-22s %12s %12s %12s" % ("CHIMERYS (PD)", fmt(pb and pb[1]), fmt(pb and pb[0]),
                                        fmt(read_pd_count(a.pd_b, None))))
    print("%-22s %12s %12s %12s" % ("우리 모델", f"{pep_ours:,}", "-", f"{len(prot):,}"))
    print()
    print("  ★ 우리는 bare 서열로 센다. PD 의 '펩타이도폼' 열과 맞대면 단위가 다르니")
    print("     **bare 서열 열이 공정한 쌍**이다. 부풀림은 인산화에서 훨씬 크다(1.2배 vs 1.04배).")
    print("  ⚠️ 우리 값은 'decoy 매칭 FDR' 이고 검색 엔진 FDR 과 **다른 양**이다.")
    print("     둘 다 1% 로 맞춰 공정성을 높였을 뿐 같다고 말하지 않는다.")
    print("  ⚠️ 참조 프로테옴 밖 서열 %s개는 개수에서 **뺐다** — 단백질에 귀속할 수 없기 때문이다."
          % f"{n_o:,}")


if __name__ == "__main__":
    main()
