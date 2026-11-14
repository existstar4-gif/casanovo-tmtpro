# -*- coding: utf-8 -*-
"""quantifiable_same_rule.py — '정량 가능한 인산화 펩타이드' 를 세 방법에 **완전히 같은 규칙**으로 센다 (2026-10-07, 사용자 요청).

왜: §15-24 의 '정량 가능 4/4 우위' 는 PD 의 Abundance 빈칸과 우리 '값>0' 을 맞댔다. PD 빈칸의 상당 부분은
공동분리 간섭 50% 초과로 PD 가 일부러 뺀 것이라(§15-25 ⑤) 같은 잣대가 아니었다.

여기서는 정량 값과 간섭 판정을 **양쪽 모두 우리 파이프라인의 스캔 단위 값**(`*.rep.tsv`: 모든 MS2 스캔의 리포터 세기와
순도)에서 가져온다. 다른 것은 '어느 스캔을 어느 펩타이드로 동정했나' 뿐이다.

  PD 쪽   = PSMs.txt 중 Confidence = High (PD 의 FDR 1%) 인산화 PSM → (stem, scan) 으로 rep 값을 붙인다
  우리 쪽 = 문턱(decoy 매칭 FDR 1%) 이상 · 참조 공간 안 인산화 PSM → 같은 rep 값
  규칙    = PSM 정량 가능 ⇔ 순도 ≥ MIN_PURITY 이고 값>0 채널 ≥ MIN_CH
            펩타이드(bare, I=L) 정량 가능 ⇔ 정량 가능한 PSM 이 하나라도 있다
  참고    = PD 자체 기준(PSM Quan Info 빈칸 = PD 가 정량에 쓴 PSM)으로 센 PD 수도 같이 낸다
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
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import count_matched_fdr as cm                 # noqa: E402
from count_matched_fdr import pp, normalize    # noqa: E402

REP_FIXED = 8      # stem scan ms2_rt prec_mz charge n_found purity ms1_scan


def bare(s):
    return normalize(re.sub(r"\[[^\]]*\]", "", s or "").replace("-", "").replace(".", "").strip().upper())


def read_rep(paths):
    """{(stem, scan): (순도, 값>0 채널 수)}"""
    out = {}
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
                pur = float(c[6])
                n = sum(1 for x in c[REP_FIXED:] if x not in ("", "nan") and float(x) > 0)
            except ValueError:
                continue
            out[(c[0], int(c[1]))] = (pur, n)
    return out


def read_pd(path):
    """PD PSMs -> [(stem, scan, bare, Quan Info 빈칸 여부)] — Confidence High · 인산화만"""
    fid = pp.load_inputfiles_map(path)
    out = []
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        hdr = [pp.clean_field(c) for c in fh.readline().rstrip("\n").split("\t")]
        ix = {n: i for i, n in enumerate(hdr)}
        cs = ix.get("Sequence")
        ann = cs is None
        if ann:
            cs = ix["Annotated Sequence"]
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if pp.clean_field(c[ix["Confidence"]]) != "High":
                continue
            if "Phospho" not in pp.clean_field(c[ix["Modifications"]]):
                continue
            stem = fid.get(pp.clean_field(c[ix["File ID"]]))
            s = pp.clean_field(c[cs])
            s = (pp.extract_peptide(s) or "") if ann else s
            try:
                scan = int(pp.clean_field(c[ix["First Scan"]]))
            except ValueError:
                continue
            qi = pp.clean_field(c[ix["Quan Info"]]) if "Quan Info" in ix else ""
            out.append((stem, scan, normalize(s.upper()), qi == ""))
    return out


def read_ours(paths):
    """[(stem, scan, bare, 점수, TMT 표지 여부)] — 인산화만"""
    out = []
    for p in paths:
        seen = False
        for line in io.open(p, encoding="utf-8"):
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[:2] == ["stem", "scan"]:
                seen = True
                continue
            if not seen or "Phospho" not in c[2]:
                continue
            try:
                sc = float(c[4])
            except ValueError:
                continue
            labeled = c[2].startswith("[TMTpro]-") and not re.search(r"K(?!\[TMTpro\])", c[2])
            out.append((c[0], int(c[1]), bare(c[2]), sc, labeled))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True, help="out/quant/<스터디> — *.rep.tsv 와 *_psm.tsv")
    ap.add_argument("--pd-psm-a", required=True)
    ap.add_argument("--pd-psm-b", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--cut", type=float, required=True, help="우리 decoy 매칭 FDR 1% 점수 문턱")
    ap.add_argument("--min-purity", type=float, default=0.5)
    ap.add_argument("--min-ch", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--label", default="study")
    a = ap.parse_args()

    rep = read_rep(sorted(glob.glob(os.path.join(a.study_dir, "*.rep.tsv"))))
    ours = [r for r in read_ours(sorted(glob.glob(os.path.join(a.study_dir, "*_psm.tsv")))) if r[3] >= a.cut]
    tgt, _dec = cm.target_decoy(a.fasta, {r[2] for r in ours})
    ours = [r for r in ours if r[2] in tgt]
    pds = {"Sequest HT": read_pd(a.pd_psm_a), "CHIMERYS": read_pd(a.pd_psm_b)}

    def count(psms, min_ch, use_rule=True):
        allp, ok = set(), set()
        miss = 0
        for stem, scan, k, flag in psms:
            allp.add(k)
            v = rep.get((stem, scan))
            if v is None:
                miss += 1
                continue
            pur, n = v
            if use_rule:
                good = pur >= a.min_purity and n >= min_ch
            else:
                good = flag
            if good:
                ok.add(k)
        return len(allp), len(ok), miss

    print("=" * 84)
    print(f"{a.label} — 정량 가능한 인산화 펩타이드, 세 방법 같은 규칙 (순도 ≥ {a.min_purity}, 값>0 채널 ≥ N)")
    print("  리포터 값·순도는 양쪽 모두 우리 파이프라인의 스캔 단위 값(*.rep.tsv)이다.")
    rows = {}
    for name, ps in pds.items():
        rows[name] = [(s, sc, k, q) for s, sc, k, q in ps]
    rows["우리 (전부)"] = [(s, sc, k, True) for s, sc, k, _x, _l in ours]
    rows["우리 (TMT 표지만)"] = [(s, sc, k, True) for s, sc, k, _x, lab in ours if lab]
    hdr = "%-18s %10s" % ("방법", "인산화 전체") + "".join("%14s" % f"채널≥{m}" for m in a.min_ch)
    print("\n" + hdr)
    base = {}
    for name, ps in rows.items():
        line = "%-18s %10s" % (name, f"{count(ps, 1)[0]:,}")
        for m in a.min_ch:
            n_all, n_ok, miss = count(ps, m)
            line += "%14s" % f"{n_ok:,}"
            base[(name, m)] = n_ok
        print(line)
    print("\n  우리 vs Sequest HT (같은 규칙):")
    for m in a.min_ch:
        s = base[("Sequest HT", m)]
        print("    채널≥%-3d  우리 전부 %+6.1f%%   TMT 표지만 %+6.1f%%   | vs CHIMERYS %+6.1f%% / %+6.1f%%" % (
            m, 100 * (base[("우리 (전부)", m)] / s - 1), 100 * (base[("우리 (TMT 표지만)", m)] / s - 1),
            100 * (base[("우리 (전부)", m)] / base[("CHIMERYS", m)] - 1),
            100 * (base[("우리 (TMT 표지만)", m)] / base[("CHIMERYS", m)] - 1)))
    print("\n  참고 — PD 자체 기준(Quan Info 빈칸 PSM 이 하나라도 있는 펩타이드):")
    for name in pds:
        n_all, n_ok, miss = count(rows[name], 0, use_rule=False)
        print("    %-12s %s / %s  (rep 에 없는 스캔 %d)" % (name, f"{n_ok:,}", f"{n_all:,}", miss))


if __name__ == "__main__":
    main()
