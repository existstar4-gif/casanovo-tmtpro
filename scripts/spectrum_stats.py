# -*- coding: utf-8 -*-
"""spectrum_stats.py — 외부 예치물별 스펙트럼 밀도 통계 (Figure 3D 의 데이터원).

원고는 "희소성이 예치물 간 정밀도 순위를 설명한다"는 해명을 **기각**하는 근거로
피크 수 중앙값을 싣는다. 그 수치가 `make_figures_v1_archive.py` 안에만 있어서
확정 시트에 등재되지 못했고, 그래서 Figure 3D 가 그려지지 않았다(2026-08-29 정리).
이 스크립트가 그 계산을 유일한 출처로 분리한다.

평가에 실제로 쓰인 스펙트럼(`sub.mgf`)만 센다. 원본 mzML 전체가 아니다.

내는 것: 스펙트럼 수 / 피크 수 중앙값 / 150 초과 비율 / 최저 m/z 중앙값
  - 150 = 모델이 읽는 최대 피크 수(`n_peaks`). 이를 넘는 스펙트럼은 잘려 들어간다.
  - 최저 m/z = 저질량 절단 예치물(PXD057861/PXD063898, 187·200 m/z)을 가려낸 그 축이다.

사용: python scripts/spectrum_stats.py [PXD054085 ...]
"""
from __future__ import annotations

import statistics
import sys

EXT = "data/external"
SIX = ["PXD065357", "PXD054085", "PXD072211", "PXD054497", "PXD063976", "PXD067164"]

# TMTpro 18채널 리포터. mzml_reporter_check.py 와 같은 값·같은 허용오차(±10 mDa)를 쓴다.
REPORTERS = [126.1277, 127.1248, 127.1311, 128.1281, 128.1344, 129.1315, 129.1378,
             130.1348, 130.1411, 131.1382, 131.1445, 132.1415, 132.1479, 133.1449,
             133.1512, 134.1482, 134.1546, 135.1516]
TOL = 0.01

# 16-plex 와 18-plex 를 가르는 깨끗한 판별자는 135N 하나뿐이다.
# 18-plex 가 추가하는 두 채널 중 134C(134.154565)는 134N(134.148245)과 6.3 mDa 차라
# ±10 mDa 허용오차 안에서 구별되지 않는다. 135N 은 1 Da 안에 이웃이 없다.
CH_135N = 135.151600
TOL_135 = 0.005


def stats(acc):
    """(n, 피크중앙값, 150초과비율, 최저m/z중앙값, 리포터중앙값) — sub.mgf 한 번만 훑는다."""
    peaks, lows, reps, cur, lo, rep = [], [], [], 0, None, 0
    n135 = 0
    has135 = False
    path = f"{EXT}/{acc}/sub.mgf"
    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return None
    with fh:
        for line in fh:
            if line.startswith("BEGIN IONS"):
                cur, lo, rep, has135 = 0, None, 0, False
            elif line.startswith("END IONS"):
                peaks.append(cur)
                reps.append(rep)
                n135 += has135
                if lo is not None:
                    lows.append(lo)
            elif line and line[0].isdigit():
                cur += 1
                m = float(line.split()[0])
                if lo is None:                    # MGF 는 m/z 오름차순 — 첫 줄이 최저
                    lo = m
                if 125.5 <= m <= 136.5 and any(abs(m - r) <= TOL for r in REPORTERS):
                    rep += 1
                if abs(m - CH_135N) <= TOL_135:
                    has135 = True
    if not peaks:
        return None
    over = 100 * sum(1 for p in peaks if p > 150) / len(peaks)
    return (len(peaks), statistics.median(peaks), over, statistics.median(lows),
            statistics.median(reps), 100 * n135 / len(peaks))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    accs = sys.argv[1:] or SIX
    print(f"{'예치물':<12}{'스펙트럼':>9}{'피크중앙':>9}{'>150':>8}{'최저m/z중앙':>12}{'리포터중앙':>11}{'135N 보유':>10}{'plex':>7}")
    for acc in accs:
        r = stats(acc)
        if r is None:
            print(f"{acc:<12}{'sub.mgf 없음':>9}")
            continue
        n, med, over, low, rep, p135 = r
        plex = "18" if p135 > 50 else ("16" if rep >= 3 else "—")
        print(f"{acc:<12}{n:>9,}{med:>9.0f}{over:>7.1f}%{low:>12.1f}{rep:>11.0f}"
              f"{p135:>9.1f}%{plex:>7}")


if __name__ == "__main__":
    main()
