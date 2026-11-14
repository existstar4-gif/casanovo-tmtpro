# -*- coding: utf-8 -*-
"""extract_reporters.py — mzML 의 MS2 마다 TMTpro 리포터 세기와 전구체 순도를 뽑는다.

M0. 정량의 바닥이다. 이 저장소에는 리포터의 **존재를 세는** 코드(`classify_tmt.py`,
`mzml_reporter_check.py`)만 있었고 **세기를 읽는** 코드가 없었다 — 정량이 통째로 비어 있었다.

두 가지를 함께 낸다. 서로 독립이지만 둘 다 있어야 정량이 믿을 만해진다.

  1. **채널 세기** — 리포터 m/z 마다 허용오차 안에서 가장 가까운 봉우리의 세기.
     봉우리가 없으면 0.0 을 쓰고, `found` 열에 몇 채널이 잡혔는지 따로 적는다.
     ⚠️ 세기 0(측정했으나 신호 없음)과 결측(측정 자체가 없음)은 다른 값이다.
        MS2 가 리포터 영역을 담지 않으면 그 행은 `n_found = -1` 로 표시해 구분한다.

  2. **전구체 순도** — MS1 의 분리 창 안에서 목표 전구체의 동위원소 봉우리가 차지하는
     세기 비율. TMT 정량의 최대 왜곡원이 공동분리이고, 순도가 낮으면 채널 비율이 1 로
     눌린다. 이 값을 PSM 마다 남겨야 나중에 필터할 수 있다.
     MS1 을 찾지 못하면 `purity = nan` 이며, 없는 것을 1.0 으로 채우지 않는다.

리포터 m/z 는 `classify_tmt.py` 와 같은 TMTpro 18채널 monoisotopic 값이고 허용오차도
같은 ±10 mDa 를 기본으로 쓴다. 두 곳이 다른 값을 쓰면 "리포터가 있다" 와
"리포터 세기가 이만큼이다" 가 어긋난다.

사용:
  python scripts/quant/extract_reporters.py in.mzML -o out/reporters.tsv
  python scripts/quant/extract_reporters.py in.mzML -o out/r.tsv --plex 16 --tol 0.01
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# TMTpro monoisotopic reporter m/z. 앞 16 개가 16-plex, 전체가 18-plex.
# classify_tmt.py 와 같은 값을 쓴다 — 두 곳이 갈리면 안 된다.
REPORTERS_18 = [
    ("126",  126.127726), ("127N", 127.124761), ("127C", 127.131081),
    ("128N", 128.128116), ("128C", 128.134436), ("129N", 129.131471),
    ("129C", 129.137790), ("130N", 130.134825), ("130C", 130.141145),
    ("131N", 131.138180), ("131C", 131.144500), ("132N", 132.141535),
    ("132C", 132.147855), ("133N", 133.144890), ("133C", 133.151210),
    ("134N", 134.148245), ("134C", 134.154565), ("135N", 135.151600),
]
PROTON = 1.007276466
C13 = 1.0033548378


def channels(plex: int):
    if plex not in (16, 18):
        raise SystemExit("plex 는 16 또는 18 이어야 한다")
    return REPORTERS_18[:16] if plex == 16 else REPORTERS_18


def scan_of(native_id, default=None):
    """'controllerType=0 controllerNumber=1 scan=123' -> 123."""
    for tok in str(native_id or "").split():
        if tok.startswith("scan="):
            return int(tok.split("=")[1])
    return default


def purity(ms1_mz, ms1_int, sel_mz, charge, center, lo_off, hi_off):
    """분리 창 안에서 목표 전구체의 동위원소가 차지하는 세기 비율.

    창은 기기가 실제로 자른 자리, 즉 분리창 target m/z 를 중심으로 둔다. selected ion
    m/z 는 나중에 단일동위원소로 보정된 값이라 창 중심과 동위원소 한 칸 이상 어긋날 수
    있다(2026-09-11 Study 1 run 에서 MS2 의 32.6%). 창 안의 총 세기를 분모로, selected ion
    m/z 부터 M+5 까지 동위원소 봉우리(±0.02 Da 안의 최대, 봉우리당 한 번)를 분자로 삼는다.
    M-1 은 넣지 않는다 — 단일동위원소보다 가벼운 봉우리는 다른 분자다.
    창에 아무것도 없으면 nan.
    """
    if ms1_mz is None or len(ms1_mz) == 0 or not charge:
        return float("nan")
    a = np.searchsorted(ms1_mz, center - lo_off)
    b = np.searchsorted(ms1_mz, center + hi_off)
    if b <= a:
        return float("nan")
    win_mz, win_int = ms1_mz[a:b], ms1_int[a:b]
    total = float(win_int.sum())
    if total <= 0:
        return float("nan")
    step = C13 / charge
    mine = 0.0
    used = set()
    for k in range(0, 6):                       # M0 .. M+5
        t = sel_mz + k * step
        i = int(np.searchsorted(win_mz, t - 0.02))
        j = int(np.searchsorted(win_mz, t + 0.02))
        if j > i:
            best = i + int(np.argmax(win_int[i:j]))
            if best not in used:
                used.add(best)
                mine += float(win_int[best])
    return min(1.0, mine / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mzml")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--plex", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.003, help="리포터 매칭 ±Da (기본 3 mDa)")
    ap.add_argument("--limit", type=int, default=0, help="0 이면 전체")
    a = ap.parse_args()

    from pyteomics import mzml as pmz

    ch = channels(a.plex)
    lo_r, hi_r = ch[0][1] - 0.5, ch[-1][1] + 0.5
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.mzml))[0]

    fh = io.open(a.out, "w", encoding="utf-8", newline="\n")
    # 무엇을 하지 않았는지를 출력에 남긴다. 이 표를 받은 사람이 보정된 값으로
    # 오해하면 정량이 조용히 틀린다.
    for _h in [
        "extract_reporters.py  plex=%d  tol=%.4f Da  source=%s" % (a.plex, a.tol, stem),
        "RAW intensities. No isotope impurity correction was applied, because no",
        "  reagent lot purity matrix was supplied. Adjacent channels may carry a",
        "  few per cent of one another's signal.",
        "No normalization of any kind. Channel loading and per-spectrum intensity",
        "  differences are left in.",
        "purity = share of the isolation window held by the intended precursor,",
        "  M0..M+5, read from the master MS1 named by the precursor spectrumRef",
        "  (ms1_scan) with the window centred on the isolation target m/z.",
        "  nan means no MS1 was available. Nothing was filtered on it.",
        "0.0 = measured, no peak within tolerance. n_found = -1 with every channel",
        "  nan = the spectrum does not cover the reporter region.",
        "Every peak goes to at most one channel, the nearest. The N and C members",
        "  of a pair are 6.32 mDa apart, so a wider tolerance lets one peak satisfy",
        "  both — that produced identical N and C columns before 2026-09-10.",
    ]:
        fh.write("# " + _h + chr(10))
    fh.write("\t".join(["stem", "scan", "ms2_rt", "prec_mz", "charge",
                        "n_found", "purity", "ms1_scan"] + [c[0] for c in ch]) + "\n")

    # 순도는 전구체를 고른 master MS1 에서 읽는다. 직전 MS1 이 아니다 — 이 기기군에서는
    # master 가 직전 MS1 보다 대개 8 스캔 앞에 있고, 직전 MS1 을 쓰면 PD 와의 일치가
    # 48% 에서 멈췄다(2026-09-11, |차이|<0.05 기준). spectrumRef 가 없는 mzML 에서만
    # 직전 MS1 로 물러난다.
    ms1_cache, ms1_order, last_ms1 = {}, [], None
    n = n_ms2 = n_norep = n_noref = n_miss = 0
    with pmz.read(a.mzml) as rd:
        for sp in rd:
            lvl = sp.get("ms level")
            if lvl == 1:
                s1 = scan_of(sp.get("id"), sp.get("index"))
                ms1_cache[s1] = (np.asarray(sp["m/z array"]), np.asarray(sp["intensity array"]))
                ms1_order.append(s1)
                if len(ms1_order) > 100:
                    ms1_cache.pop(ms1_order.pop(0), None)
                last_ms1 = s1
                continue
            if lvl != 2:
                continue
            n_ms2 += 1
            mz = np.asarray(sp["m/z array"])
            it = np.asarray(sp["intensity array"])
            scan = scan_of(sp.get("id"), sp.get("index", n_ms2))
            prec = sp.get("precursorList", {}).get("precursor", [{}])[0]
            sel = prec.get("selectedIonList", {}).get("selectedIon", [{}])[0]
            pmzv = float(sel.get("selected ion m/z", 0.0) or 0.0)
            zv = int(sel.get("charge state", 0) or 0)
            win = prec.get("isolationWindow", {})
            center = float(win.get("isolation window target m/z", pmzv) or pmzv)
            lo_off = float(win.get("isolation window lower offset", 0.7) or 0.7)
            hi_off = float(win.get("isolation window upper offset", lo_off) or lo_off)
            ref = scan_of(prec.get("spectrumRef"))
            if ref is None:
                n_noref += 1
                ref = last_ms1

            # 스펙트럼이 리포터 영역을 담고 있는가 — 없으면 결측으로 표시한다
            if len(mz) == 0 or mz[0] > hi_r or mz[-1] < lo_r:
                n_norep += 1
                fh.write("\t".join([stem, str(scan),
                                    "%.4f" % float(sp.get("scanList", {}).get("scan", [{}])[0]
                                                   .get("scan start time", 0.0) or 0.0),
                                    "%.5f" % pmzv, str(zv), "-1", "nan", str(ref)]
                                   + ["nan"] * len(ch)) + "\n")
                continue

            # 한 봉우리를 두 채널이 나눠 갖지 않게 한다. N/C 짝은 6.32 mDa 밖에
            # 떨어져 있지 않아, 허용오차가 그 절반을 넘으면 같은 봉우리가 두 채널을
            # 모두 만족한다. tol=10 mDa 에서 127N 과 127C 가 전 행에서 같은 값을
            # 냈다(2026-09-10 실측). 봉우리마다 가장 가까운 채널 하나에만 준다.
            vals = [0.0] * len(ch)
            claimed = {}
            for ci, (_nm, t) in enumerate(ch):
                lo = int(np.searchsorted(mz, t - a.tol))
                hi = int(np.searchsorted(mz, t + a.tol))
                for j in range(lo, hi):
                    d = abs(float(mz[j]) - t)
                    prev = claimed.get(j)
                    if prev is None or d < prev[1]:
                        claimed[j] = (ci, d)
            for j, (ci, _d) in claimed.items():
                if float(it[j]) > vals[ci]:
                    vals[ci] = float(it[j])
            found = sum(1 for v in vals if v > 0.0)
            if ref in ms1_cache:
                pur = purity(*ms1_cache[ref], pmzv, zv, center, lo_off, hi_off)
            else:
                n_miss += 1
                pur = float("nan")
            rt = float(sp.get("scanList", {}).get("scan", [{}])[0].get("scan start time", 0.0) or 0.0)
            fh.write("\t".join([stem, str(scan), "%.4f" % rt, "%.5f" % pmzv, str(zv),
                                str(found), "nan" if math.isnan(pur) else "%.4f" % pur, str(ref)]
                               + ["%.1f" % v for v in vals]) + "\n")
            n += 1
            if a.limit and n >= a.limit:
                break
    fh.close()
    print("MS2 %d개 중 %d개 기록, 리포터 영역 없음 %d개 -> %s" % (n_ms2, n, n_norep, a.out))
    print("  spectrumRef 없음(직전 MS1 사용) %d개, master MS1 을 찾지 못함(purity nan) %d개"
          % (n_noref, n_miss))


if __name__ == "__main__":
    main()
