# -*- coding: utf-8 -*-
"""combine_psm_quant.py — M2. de novo 동정과 리포터 정량을 (stem, scan) 으로 붙인다.

M0 이 리포터 세기를, M1 이 순도를 냈고 둘 다 **서열을 모르는 채** 스펙트럼 단위로만 존재했다.
이 단계가 거기에 서열을 붙여 펩타이드 수준 정량표를 만든다. `QUANT_PLAN.md` §2 의 [C] 다.

조인 키는 **(파일 stem, scan)** 이다. scan 만으로 잡으면 파일 간 scan 번호가 겹쳐 라벨이
덮인다 — `rescue_unified.py` 규칙 1 과 같은 이유이고, 실제로 그 사고가 한 번 있었다.

지키는 것 (`QUANT_PLAN.md` §3):
  · 정규화하지 않는다. 채널 투입량 보정도 비율 계산도 하지 않는다 — 하류의 몫이다.
  · **세기 0 과 결측을 구분한다.** 리포터 영역이 없는 스펙트럼(n_found = -1)은 채널이 nan
    이고 집계에서 **빠진다**. 0 으로 채우면 없는 측정을 있다고 말하는 것이다.
  · 순도는 항상 낸다. 거르는 것은 기본값이 아니다(2026-09-11 사용자 결정).
  · 점수 임계도 기본값 없음. 거르면 헤더에 적는다.
  · 트립신 공간 밖 서열은 **버리지 않고 표시만 한다**(`in_db`). 단백질 귀속에서 빼는 것은
    M3 의 일이고, 여기서 버리면 우리가 건진 것을 우리가 지우는 셈이다.
  · `in_db` 는 FDR 이 아니다. 검색엔진 FDR 과 다른 양이며 그렇게 부르지 않는다.
  · 모든 설정을 출력 헤더에 각인한다.

사용:
  python scripts/quant/combine_psm_quant.py --mztab p.mztab --mgf run.mgf \\
      --reporters out/quant/rep_example.tsv --out-prefix out/quant/example \\
      --index out/tryptic_mouse_v2.npy
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "scripts"))
import build_tryptic_index as bti      # noqa: E402  — 트립신 공간 정의는 여기 하나뿐
import rescue_unified as ru            # noqa: E402  — mzTab+MGF 파서를 공유한다

FIXED = ("stem", "scan", "ms2_rt", "prec_mz", "charge", "n_found", "purity", "ms1_scan")


def read_reporters(path):
    """extract_reporters.py 출력을 읽는다 -> (채널이름들, {(stem, scan): 행})."""
    rows, chans = {}, None
    for line in io.open(path, encoding="utf-8"):
        if line.startswith("#"):
            continue
        c = line.rstrip("\n").split("\t")
        if chans is None:                       # 헤더 줄
            if c[:2] != ["stem", "scan"]:
                raise SystemExit("리포터 표의 헤더가 예상과 다르다: " + path)
            chans = c[len(FIXED):]
            continue
        d = dict(zip(FIXED, c[:len(FIXED)]))
        d["vals"] = c[len(FIXED):]
        rows[(d["stem"], int(d["scan"]))] = d
    if chans is None:
        raise SystemExit("리포터 표가 비어 있다: " + path)
    return chans, rows


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def in_db_lookup(index_path):
    """트립신 공간 조회기. 인덱스가 없으면 항상 None 을 돌려준다(판정 안 함)."""
    if not index_path:
        return lambda pep: None
    arr = np.load(index_path)

    def hit(pep):
        h = np.uint64(bti.h64(bti.norm(pep)))
        i = int(np.searchsorted(arr, h))
        return bool(i < len(arr) and arr[i] == h)
    return hit


def stamp(fh, a, chans, n_rep, note):
    fh.write("# combine_psm_quant.py (M2)  mztab=%s  reporters=%s\n"
             % (os.path.basename(a.mztab), os.path.basename(a.reporters)))
    fh.write("# join key = (stem, scan).  %d channels: %s\n" % (len(chans), ",".join(chans)))
    fh.write("# RAW intensities carried through unchanged. No normalization, no ratios,\n"
             "#   no isotope impurity correction beyond whatever M0 already applied.\n")
    fh.write("# score filter: %s   purity filter: %s   aggregation: %s\n"
             % ("none" if a.min_score is None else ">= %g" % a.min_score,
                "none" if a.min_purity is None else ">= %g" % a.min_purity,
                a.agg))
    fh.write("# in_db = the peptide is present in a tryptic digest of the reference proteome\n"
             "#   (%s). It is NOT a false discovery rate and must not be reported as one.\n"
             % (os.path.basename(a.index) if a.index else "no index supplied — column empty"))
    fh.write("# Spectra whose MS2 does not cover the reporter region (n_found = -1) carry nan\n"
             "#   channels and are EXCLUDED from aggregation; 0.0 means measured with no peak.\n")
    fh.write("# reporter rows read: %d\n" % n_rep)
    if note:
        fh.write("# %s\n" % note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mztab", required=True, help="de novo 예측 mzTab")
    ap.add_argument("--mgf", required=True, help="그 예측을 낸 MGF (spectra_ref index 복원용)")
    ap.add_argument("--reporters", required=True, help="extract_reporters.py 출력 TSV")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--index", default=None, help="트립신 공간 npy (없으면 in_db 판정 안 함)")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--min-purity", type=float, default=None)
    ap.add_argument("--agg", choices=("sum", "median"), default="sum",
                    help="펩타이드 집계 방식. 기본 sum 은 PD/FragPipe 관행과 같다")
    ap.add_argument("--keep-stem", default=None, help="이 stem 만 처리")
    a = ap.parse_args()

    chans, rep = read_reporters(a.reporters)
    print("리포터 표 %d행, 채널 %d개" % (len(rep), len(chans)))

    rows, n_spec = ru.load(a.mztab, a.mgf)
    if a.keep_stem:
        rows = [r for r in rows if r[0] == a.keep_stem]
    print("mzTab 예측 %d건 (MGF 스펙트럼 %d개)" % (len(rows), n_spec))

    hit = in_db_lookup(a.index)

    # ── PSM 표 ────────────────────────────────────────────────────────────
    psm_path = a.out_prefix + "_psm.tsv"
    n_join = n_nojoin = n_score = n_pur = n_norep = 0
    kept = []
    with io.open(psm_path, "w", encoding="utf-8", newline="") as fh:
        stamp(fh, a, chans, len(rep), None)
        fh.write("\t".join(["stem", "scan", "peptidoform", "sequence", "score", "ppm",
                            "charge", "purity", "n_found", "in_db"] + chans) + "\n")
        for stem, scan, pro, sc, ppm in rows:
            r = rep.get((stem, scan))
            if r is None:
                n_nojoin += 1
                continue
            n_join += 1
            if a.min_score is not None and sc < a.min_score:
                n_score += 1
                continue
            pur = fnum(r["purity"])
            if a.min_purity is not None and not (pur >= a.min_purity):
                n_pur += 1
                continue
            nf = int(r["n_found"])
            if nf < 0:
                n_norep += 1
            seq = ru.bare(pro)
            db = hit(seq)
            fh.write("\t".join([stem, str(scan), pro, seq, "%.6f" % sc, "%.2f" % ppm,
                                r["charge"],
                                "nan" if math.isnan(pur) else "%.4f" % pur,
                                str(nf), "" if db is None else ("1" if db else "0")]
                               + r["vals"]) + "\n")
            kept.append((pro, seq, sc, pur, nf, r["vals"], db))
    print("  조인 성공 %d / 리포터 행 없음 %d" % (n_join, n_nojoin))
    if a.min_score is not None:
        print("  점수 미달로 제외 %d" % n_score)
    if a.min_purity is not None:
        print("  순도 미달로 제외 %d" % n_pur)
    print("  리포터 영역 없는 스펙트럼 %d (집계에서 제외)" % n_norep)
    print("  -> %s" % psm_path)

    # ── 펩타이드 표 ───────────────────────────────────────────────────────
    agg = {}
    for pro, seq, sc, pur, nf, vals, db in kept:
        g = agg.setdefault(pro, {"seq": seq, "n": 0, "nq": 0, "norep": 0, "zero": 0,
                                 "best": -1e18, "pur": [], "cols": [[] for _ in chans],
                                 "db": db})
        g["n"] += 1
        g["best"] = max(g["best"], sc)
        if not math.isnan(pur):
            g["pur"].append(pur)
        if nf < 0:                       # 측정 자체가 없다 — 집계에 넣지 않는다
            g["norep"] += 1
            continue
        v = [fnum(x) for x in vals]
        if any(math.isnan(x) for x in v):
            g["norep"] += 1
            continue
        g["nq"] += 1
        if not any(x > 0 for x in v):
            g["zero"] += 1
        for i, x in enumerate(v):
            g["cols"][i].append(x)

    pep_path = a.out_prefix + "_peptide.tsv"
    with io.open(pep_path, "w", encoding="utf-8", newline="") as fh:
        stamp(fh, a, chans, len(rep),
              "peptide level: one row per peptidoform. n_psm_quant is the number of PSMs that "
              "actually carried reporter values and is the only count the channels are built from.")
        fh.write("\t".join(["peptidoform", "sequence", "n_psm", "n_psm_quant", "n_psm_noreporter",
                            "n_psm_allzero", "best_score", "median_purity", "in_db"] + chans) + "\n")
        for pro in sorted(agg, key=lambda k: -agg[k]["n"]):
            g = agg[pro]
            if g["nq"] == 0:
                vals = ["nan"] * len(chans)
            elif a.agg == "sum":
                vals = ["%.1f" % sum(c) for c in g["cols"]]
            else:
                vals = ["%.1f" % float(np.median(c)) for c in g["cols"]]
            mp = float(np.median(g["pur"])) if g["pur"] else float("nan")
            fh.write("\t".join([pro, g["seq"], str(g["n"]), str(g["nq"]), str(g["norep"]),
                                str(g["zero"]), "%.6f" % g["best"],
                                "nan" if math.isnan(mp) else "%.4f" % mp,
                                "" if g["db"] is None else ("1" if g["db"] else "0")]
                               + vals) + "\n")

    nq = sum(1 for g in agg.values() if g["nq"] > 0)
    indb = sum(1 for g in agg.values() if g["db"])
    print("펩타이도폼 %d개 (정량값 있는 것 %d개)" % (len(agg), nq))
    if a.index:
        print("  트립신 공간 안 %d개 (%.1f%%)" % (indb, 100 * indb / len(agg) if agg else 0))
    print("  -> %s" % pep_path)


if __name__ == "__main__":
    main()
