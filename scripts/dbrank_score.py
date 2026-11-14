# -*- coding: utf-8 -*-
"""dbrank_score.py — 트립신 공간 재순위 3팔(A/B/C)을 외부 데이터셋에서 채점한다.

채점 규칙은 make_figures.py::score_external 을 그대로 재사용한다. 특히 GT 키는
반드시 (파일 stem, scan) 이다 — scan 만으로 잡으면 PXD054085 처럼 두 raw 파일이
scan 번호를 24,274개 공유하는 데이터에서 정답이 덮여 정밀도가 87.4 -> 60.2% 로
잘못 내려간다(2026-08-28 발견).

세 팔은 후보를 버리지 않고 재정렬만 하므로 커버리지가 같아야 한다. 그것도 함께 검증한다.

사용: python scripts/dbrank_score.py
"""
from __future__ import annotations

import importlib.util
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = "data/external"
OUT = "data/eval/dbrank_ext"

_spec = importlib.util.spec_from_file_location(
    "mf_", os.path.join(BASE, "scripts", "make_figures.py"))
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

SETS = [("PXD054085", "global, mouse"), ("PXD065357", "label-free, human"),
        ("PXD072211", "global, mouse"), ("PXD063976", "phospho, human"),
        ("PXD054497", "pY, mouse"), ("PXD067164", "pY, human")]


def keys_of(mgf):
    out = []
    for line in io.open(mgf, encoding="utf-8", errors="ignore"):
        if line.startswith("TITLE="):
            st, sc = line.strip()[6:].rsplit(":scan:", 1)
            out.append((st, int(sc)))
    return out


def score(acc, arm):
    mz = f"{OUT}/{acc}_{arm}.mztab"
    mgf = f"{EXT}/{acc}/sub.mgf"
    if not (os.path.exists(mz) and os.path.exists(mgf)):
        return None
    gt = {}
    for line in io.open(f"{EXT}/{acc}/gt.tsv", encoding="utf-8").read().splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 5:
            gt[(c[0], int(c[1]))] = c[4]
    raw = mf.load_mztab(mz)
    ok = n = cov = 0
    for i, k in enumerate(keys_of(mgf)):
        g = gt.get(k)
        if not g:
            continue
        p = raw.get(i, raw.get(k[1], ("", 0)))[0]
        n += 1
        cov += bool(p)
        ok += bool(p) and mf.full(p) == mf.full(g)
    return ok, n, cov


def main():
    print(f"{'데이터셋':<12}{'구성':<20}{'n':>6}"
          f"{'A 없음':>9}{'B 질량':>9}{'C 질량+DB':>11}{'B→C':>8}   커버리지")
    rows = []
    for acc, desc in SETS:
        r = {a: score(acc, a) for a in "ABC"}
        if any(v is None for v in r.values()):
            miss = [a for a in "ABC" if r[a] is None]
            print(f"{acc:<12}{desc:<20}  (미완: {','.join(miss)})")
            continue
        n = r["B"][1]
        p = {a: 100 * r[a][0] / r[a][1] for a in "ABC"}
        cov = {a: r[a][2] for a in "ABC"}
        same = "동일" if cov["A"] == cov["B"] == cov["C"] else f"다름 {cov}"
        print(f"{acc:<12}{desc:<20}{n:>6}{p['A']:>8.1f}%{p['B']:>8.1f}%"
              f"{p['C']:>10.1f}%{p['C']-p['B']:>+7.1f}   {same}")
        rows.append((acc, n, p["A"], p["B"], p["C"]))
    if rows:
        tot = sum(r[1] for r in rows)
        w = {k: sum(r[1] * r[i] for r in rows) / tot for i, k in ((2, "A"), (3, "B"), (4, "C"))}
        print(f"\n{'가중 평균':<32}{tot:>6}{w['A']:>8.1f}%{w['B']:>8.1f}%"
              f"{w['C']:>10.1f}%{w['C']-w['B']:>+7.1f}")


if __name__ == "__main__":
    main()
