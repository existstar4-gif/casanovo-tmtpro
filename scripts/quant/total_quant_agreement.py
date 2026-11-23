# -*- coding: utf-8 -*-
"""total_quant_agreement.py — global proteome 스터디의 TMT 정량 일치 (2026-11-17).

global proteome 은 단백질 기준 비교가 맞다 (인산화의 '같은 자리끼리' 규칙은 인산화에만 해당).
펩타이드 프로파일 = 채널별 리포터 합 → log2(채널/채널 평균). 두 프로파일의 Pearson r 로 잰다.

  A 같은 펩타이드: 우리와 PD 가 둘 다 찾은 펩타이드의 프로파일 일치 (PSM 집합이 달라도 같은 값을 내는가)
  B 단백질 일치  : 펩타이드 프로파일 vs 같은 단백질의 PD(CHIMERYS) 단백질 프로파일
                   (그 펩타이드를 뺀 나머지 PD 펩타이드 합, PD 펩타이드 3개 이상인 단백질만)
                   집단 = 공유 / 우리만(Sequest·CHIMERYS 둘 다 없음) / 무작위 단백질(기저)
                   π = (우리만 − 무작위) / (공유 − 무작위)
채널 정규화 (11-17 추가): 채널 적재량 차이가 모든 펩타이드에 공통으로 실리면(Study 3: 무작위 단백질 r 0.950 ≈ 공유 0.953)
상관이 판별력을 잃는다. 그래서 PD(CHIMERYS) 펩타이드 프로파일의 채널별 중앙값을 모든 프로파일에서 뺀다 (median normalization).
PSM 정량 조건(양쪽 동일): 값>0 채널 ≥ 8. 우리 쪽은 compare_total.py 의 FDR 1% 문턱 이상 · N말단 표지 · 참조 공간 안.

    python scripts/quant/total_quant_agreement.py --study study3t
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_phospho_split as cps                    # noqa: E402
from count_matched_fdr import target_decoy, pp, normalize   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(HERE))
FA = os.path.join(BASE, "out/fasta/mouse_sp_plus_trembl_missing_20260708.fasta")
PD = {"study3t": "data/pd/study3_total_fr_",
      "study2t": "data/pd/study2_total_fr_",
      "study1t": "data/pd/study1_total_fr_"}
LABEL = {"study1t": "Study 1", "study2t": "Study 2", "study3t": "Study 3"}
MIN_CH = 8


NORM = None   # 채널별 중앙값 (main 에서 PD 펩타이드로 잡는다)
KEEP = None   # 빈 채널(중앙값보다 8배 넘게 약한 채널, Study 3 의 126–128N)을 뺀 채널 마스크


def profile(v):
    v = np.asarray(v, float)
    if KEEP is not None:
        v = v[KEEP]
    if (v > 0).sum() < MIN_CH:
        return None
    v = np.log2(np.maximum(v, 1.0))
    v = v - v.mean()
    return v - NORM if NORM is not None else v


def r(a, b):
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return np.nan
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def read_ours(study, cut):
    sums, nch = {}, None
    for p in sorted(glob.glob(os.path.join(BASE, "out/quant", study, "*_psm.tsv"))):
        hdr = None
        for line in io.open(p, encoding="utf-8"):
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if hdr is None:
                hdr = c; i0 = hdr.index("in_db") + 1; nch = len(hdr) - i0
                continue
            if not c[2].startswith("[") or float(c[4]) < cut:
                continue
            v = np.array(c[i0:], float)
            if (v > 0).sum() < MIN_CH:
                continue
            k = cps.bare(c[2])
            sums[k] = sums.get(k, 0) + v
    return sums, nch


def read_pd(path, nch):
    sums, prot = {}, {}
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        hdr = [pp.clean_field(x) for x in fh.readline().rstrip("\n").split("\t")]
        ix = {x: i for i, x in enumerate(hdr)}
        ab = [i for i, x in enumerate(hdr) if x.startswith("Abundance: ")][:nch]
        for line in fh:
            c = [pp.clean_field(x) for x in line.rstrip("\n").split("\t")]
            if c[ix["Confidence"]] != "High":
                continue
            s = pp.extract_peptide(c[ix["Annotated Sequence"]]) or ""
            if not s:
                continue
            k = normalize(s.upper())
            v = np.array([float(c[i]) if c[i] else 0.0 for i in ab])
            accs = [x.strip() for x in c[ix["Master Protein Accessions"]].split(";") if x.strip()]
            if len(accs) == 1:
                prot[k] = accs[0]
            if (v > 0).sum() < MIN_CH:
                continue
            sums[k] = sums.get(k, 0) + v
    return sums, prot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=list(PD))
    ap.add_argument("--json")
    a = ap.parse_args()
    cut = json.load(open(os.path.join(BASE, f"out/quant/total_compare_{a.study}.json")))["threshold"]
    ours, nch = read_ours(a.study, cut)
    tgt, _ = target_decoy(FA, set(ours))
    ours = {k: v for k, v in ours.items() if k in tgt}
    seq, _ = read_pd(PD[a.study] + "_PSMs.txt", nch)
    chi, chi_prot = read_pd(PD[a.study] + "-(1)_PSMs.txt", nch)

    global NORM, KEEP
    raw = [p for p in (profile(v) for v in chi.values()) if p is not None]
    off = np.median(np.vstack(raw), axis=0)
    KEEP = off - np.median(off) > -3          # log2: 8배 넘게 약하면 빈 채널
    raw = [p for p in (profile(v) for v in chi.values()) if p is not None]
    NORM = np.median(np.vstack(raw), axis=0)

    print("=" * 84)
    print(f"{LABEL[a.study]} global proteome — TMT quantitative agreement (channels {nch}, PSM >= {MIN_CH} non-zero channels, channel median-normalized)")
    print("  channel offsets before exclusion (log2): " + " ".join(f"{x:+.2f}" for x in off)
          + f"  -> {int((~KEEP).sum())} empty channel(s) excluded")
    res = {}
    # A 같은 펩타이드
    for name, pdset in (("Sequest HT", seq), ("CHIMERYS", chi)):
        rs = []
        for k in ours.keys() & pdset.keys():
            pa, pb = profile(ours[k]), profile(pdset[k])
            if pa is not None and pb is not None:
                rs.append(r(pa, pb))
        rs = np.array([x for x in rs if x == x])
        print(f"  A same peptide vs {name:<10}: n={len(rs):,}  median r={np.median(rs):.3f}  r>=0.9 {100 * (rs >= 0.9).mean():.1f}%")
        res[f"A_{name}"] = dict(n=len(rs), median_r=float(np.median(rs)), frac_r_ge_0_9=float((rs >= 0.9).mean()))

    # B 단백질 일치 — PD(CHIMERYS) 단백질 프로파일
    members = collections.defaultdict(list)
    for k, acc in chi_prot.items():
        if k in chi:
            members[acc].append(k)
    ptot = {acc: sum(chi[k] for k in ks) for acc, ks in members.items() if len(ks) >= 3}
    def prot_r(k, vec, acc):
        if acc not in ptot:
            return None
        ref = ptot[acc] - chi[k] if k in chi else ptot[acc]
        pa, pb = profile(vec), profile(ref)
        return r(pa, pb) if pa is not None and pb is not None else None
    either = seq.keys() | chi.keys()
    groups = {"shared": [], "ours only": [], "random protein": []}
    accs = list(ptot)
    rnd = random.Random(454)
    for k, v in ours.items():
        hits = tgt.get(k, ())
        if len(hits) != 1:
            continue
        acc = hits[0]
        g = "shared" if k in either else "ours only"
        x = prot_r(k, v, acc)
        if x is not None and x == x:
            groups[g].append(x)
            y = prot_r(k, v, rnd.choice(accs))
            if y is not None and y == y:
                groups["random protein"].append(y)
    med = {g: float(np.median(v)) if v else float("nan") for g, v in groups.items()}
    for g, v in groups.items():
        print(f"  B {g:<15} vs PD protein: n={len(v):,}  median r={med[g]:.3f}")
    pi = (med["ours only"] - med["random protein"]) / (med["shared"] - med["random protein"])
    print(f"  π (ours only relative to shared, random = 0) = {pi:.2f}")
    res["B"] = {g: dict(n=len(v), median_r=med[g]) for g, v in groups.items()}
    res["B"]["pi"] = pi
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
