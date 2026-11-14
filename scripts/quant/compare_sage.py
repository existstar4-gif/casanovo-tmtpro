# -*- coding: utf-8 -*-
"""compare_sage.py — 독립 검색 엔진 Sage 의 판정을 우리 PSM 과 스캔 단위로 맞댄다 (2026-10-06, 사용자 요청).

Sage(v0.14.7, lazear/sage)를 PD 와 같은 설정으로 돌렸다: TMTpro N말단·K 고정, C 카바미도메틸 고정, 인산 STY·산화 M 가변
(최대 2개 — 3개는 색인이 메모리를 넘었다), 누락 절단 2, 전구체 10 ppm, 조각 20 ppm, 내부 decoy, report_psms 1.

스캔마다 Sage 1등과 우리 예측을 비교해 네 갈래로 나눈다.
  ① 같은 서열, Sage spectrum q ≤ 0.01   = 독립 엔진이 확인
  ② 같은 서열, q > 0.01                  = 같은 방향, 근거 약함
  ③ 다른 서열(target)                    = 우리 답이 틀렸을 가능성
  ④ Sage 보고 없음 / decoy 1등           = 판단 불가
Sage 가 구조적으로 찾을 수 없는 우리 예측(미표지 = TMT 없음, 가변 수식 3개 이상, 아세틸 등 설정 밖 수식)은 따로 센다.
같은 서열이면 인산화 자리까지 같은지도 본다.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
csv.field_size_limit(10 ** 9)


def bare(pf):
    return re.sub(r"\[[^\]]*\]", "", pf).replace("-", "").upper().replace("I", "L")


def phospho_sites(pf):
    """'[TMTpro]-AS[Phospho]K' 또는 'AS[+79.9663]K' -> 인산 자리 집합(0부터, 잔기 기준)"""
    s = re.sub(r"^(\[[^\]]+\]-?)+", "", pf)
    sites, i = set(), -1
    for m in re.finditer(r"([A-Z])((?:\[[^\]]+\])*)", s):
        i += 1
        if "Phospho" in m.group(2) or "+79.96" in m.group(2):
            sites.add(i)
    return sites


def searchable(pf):
    """Sage 설정으로 찾을 수 있는 예측인가 — (가능, 이유)"""
    if not pf.startswith("[TMTpro]-"):
        return False, "미표지(N말단 TMT 없음)"
    if re.search(r"K(?!\[TMTpro\])", pf):
        return False, "K 미표지"
    names = re.findall(r"\[([^\]]+)\]", pf)
    if any(n not in ("TMTpro", "Phospho", "Oxidation", "Carbamidomethyl") for n in names):
        return False, "설정 밖 수식"
    if names.count("Phospho") + names.count("Oxidation") > 2:
        return False, "가변 수식 3개 이상"
    return True, ""


def read_sage(path):
    out = {}
    with io.open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            stem = os.path.splitext(os.path.basename(r["filename"]))[0]
            m = re.search(r"scan=(\d+)", r["scannr"])
            if not m:
                continue
            out[(stem, int(m.group(1)))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sage", required=True, help="results.sage.tsv")
    ap.add_argument("--metrics", required=True, help="annotate_psm.py 의 psm_metrics.tsv")
    ap.add_argument("--filter", default=None, help="validate_filter.py 의 filter.tsv (있으면 교차표)")
    ap.add_argument("--compete", default=None, help="compete_candidates.py 의 compete.tsv (있으면 교차표)")
    ap.add_argument("--label", default="study")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sage = read_sage(a.sage)
    filt = {}
    if a.filter:
        for r in csv.DictReader(io.open(a.filter, encoding="utf-8"), delimiter="\t"):
            filt[(r["stem"], int(r["scan"]))] = r["passed"] == "1"
    comp = {}
    if a.compete:
        for r in csv.DictReader(io.open(a.compete, encoding="utf-8"), delimiter="\t"):
            comp[(r["stem"], int(r["scan"]))] = r["beaten"] == "1"

    cnt = collections.defaultdict(collections.Counter)
    site = collections.defaultdict(collections.Counter)
    cross = collections.defaultdict(collections.Counter)
    rows = []
    for r in csv.DictReader(io.open(a.metrics, encoding="utf-8"), delimiter="\t"):
        cat, key, pf = r["category"], (r["stem"], int(r["scan"])), r["peptidoform"]
        ok, why = searchable(pf)
        if not ok:
            cnt[cat]["제외: " + why] += 1
            continue
        cnt[cat]["n"] += 1
        s = sage.get(key)
        if s is None or s["label"] != "1":
            v = "④ 보고 없음/decoy"
        elif bare(s["peptide"]) == bare(pf):
            v = "① 같은 서열, q≤1%" if float(s["spectrum_q"]) <= 0.01 else "② 같은 서열, q>1%"
            site[cat]["same"] += phospho_sites(s["peptide"]) == phospho_sites(pf)
            site[cat]["n"] += 1
        else:
            v = "③ 다른 서열"
        cnt[cat][v] += 1
        if key in filt and key in comp:
            grp = "필터 통과 & 안 짐" if (filt[key] and not comp[key]) else ("다른 서열이 이김" if comp[key] else "필터 탈락")
            cross[(cat, grp)][v] += 1
        rows.append((cat, r["stem"], r["scan"], pf, v, s["peptide"] if s else "", s["spectrum_q"] if s else ""))

    cats = ("세 방법 공통", "부분 겹침", "우리만")
    verdicts = ("① 같은 서열, q≤1%", "② 같은 서열, q>1%", "③ 다른 서열", "④ 보고 없음/decoy")
    print("=" * 84)
    print(f"{a.label} — Sage(독립 검색 엔진) 판정 vs 우리 예측, 스캔 단위")
    print("%-12s %7s  " % ("범주", "PSM") + "  ".join("%-16s" % v for v in verdicts) + "  자리일치")
    for cat in cats:
        c = cnt[cat]
        n = c["n"]
        if not n:
            continue
        sc = site[cat]
        print("%-12s %7s  " % (cat, f"{n:,}") + "  ".join("%15.1f%%" % (100 * c[v] / n) for v in verdicts)
              + "   %5.1f%%" % (100 * sc["same"] / max(1, sc["n"])))
    print("\n  Sage 로 찾을 수 없어 뺀 예측:")
    for cat in cats:
        ex = {k: v for k, v in cnt[cat].items() if k.startswith("제외")}
        print("    %-12s %s" % (cat, ", ".join(f"{k[4:]} {v:,}" for k, v in sorted(ex.items())) or "-"))
    if cross:
        print("\n── 우리 두 검사(근거 필터·후보 경쟁) 판정별 Sage 결과 (경쟁 표본 800 개 범위) ──")
        for cat in cats:
            for grp in ("필터 통과 & 안 짐", "필터 탈락", "다른 서열이 이김"):
                c = cross.get((cat, grp))
                if not c:
                    continue
                n = sum(c.values())
                print("  %-12s %-14s n=%4d  " % (cat, grp, n)
                      + "  ".join("%s %5.1f%%" % (v[0], 100 * c[v] / n) for v in verdicts))
    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as fh:
            fh.write("category\tstem\tscan\tours\tverdict\tsage_peptide\tsage_spectrum_q\n")
            for row in rows:
                fh.write("\t".join(str(x) for x in row) + "\n")
        print("  ->", a.out)


if __name__ == "__main__":
    main()
