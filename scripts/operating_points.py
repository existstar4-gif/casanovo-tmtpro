# -*- coding: utf-8 -*-
"""operating_points.py — 오류율을 고정한 운영점 표 (배포 가중치를 쓰려면 이게 있어야 한다).

de novo 결과는 임계값 없이 쓸 수 없다. 원고는 커버리지 100% 단일 지점의 정밀도만 싣고
있어서, 독자가 "오류 5% 를 감수하면 몇 %를 얻나"에 답할 수 없다(2026-08-30 적대 검토 지적).
이 스크립트가 개발셋(TMT / 무표지 분리)과 외부 6건에 대해 그 표를 만든다.

정의:
  · 예측을 점수 내림차순으로 정렬하고, 상위 k 개만 받아들이는 운영점을 훑는다.
  · 커버리지 = k / 평가 스펙트럼 전량(시퀀싱 실패도 분모에 남는다 — 원고의 정밀도 정의와 동일).
  · 정밀도 = 받아들인 k 개 중 펩타이도폼 완전 일치 비율.
  · 보고값 = 정밀도 >= 임계(99% / 95%)를 만족하는 **최대 커버리지**와 그때의 점수 문턱.
  · 점수는 casanovo 의 search_engine_score[1] 원값이다. 길이 정규화는 하지 않는다
    (길이 계층이 정밀도와 얽혀 있어 정규화가 층을 섞는다 — 정규화판은 --norm 으로 확인 가능).

사용: python scripts/operating_points.py [--norm]
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = "data/external"
SIX = ["PXD054085", "PXD065357", "PXD072211", "PXD063976", "PXD054497", "PXD067164"]
MOD = re.compile(r"\[[^\]]*\]")
NORM = "--norm" in sys.argv


def _mf():
    import matplotlib
    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location("mf", os.path.join(BASE, "scripts", "make_figures.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def key(score, seq):
    """--norm 이면 길이로 기하평균화한 점수를 쓴다."""
    if not NORM or not seq:
        return score
    n = max(1, len(MOD.sub("", seq).replace("-", "")))
    return score / n


def sweep(rows, n_total):
    """rows = [(score, ok)] — 정밀도 임계별 최대 커버리지와 점수 문턱."""
    v = sorted(rows, key=lambda r: -r[0])
    out, ok = {}, 0
    for i, (sc, hit) in enumerate(v, 1):
        ok += hit
        prec = 100 * ok / i
        for th in (99.0, 95.0, 90.0):
            if prec >= th:
                out[th] = (100 * i / n_total, prec, sc)
    return out


def fmt(d, th):
    if th not in d:
        return f"{'—':>18}"
    cov, prec, sc = d[th]
    return f"{cov:>6.1f}% @ s>{sc:>6.2f}"


def main():
    mf = _mf()
    print(f"점수: {'길이 정규화(score/L)' if NORM else 'casanovo 원값'}\n")
    print(f"{'평가셋':<22}{'n':>7}{'전량 정밀도':>12}"
          f"{'정밀도>=99%':>20}{'정밀도>=95%':>20}{'정밀도>=90%':>20}")

    rec = mf.score_dev("data/eval/final_knap.mztab")
    for cls, lab in (("tmt", "개발셋 TMT"), ("naked", "개발셋 label-free")):
        v = [r for r in rec if r["cls"] == cls]
        rows = [(key(r["score"], r["pred"]), r["ok"]) for r in v]
        d = sweep(rows, len(v))
        base = 100 * sum(r["ok"] for r in v) / len(v)
        print(f"{lab:<22}{len(v):>7,}{base:>11.1f}%{fmt(d,99.0):>20}{fmt(d,95.0):>20}{fmt(d,90.0):>20}")

    for acc in SIX:
        d_ = f"{EXT}/{acc}"
        mzt = f"{d_}/ext.mztab"
        if not os.path.exists(mzt):
            mzt = f"{d_}/ext_knap.mztab"
        if not (os.path.exists(mzt) and os.path.exists(f"{d_}/sub.mgf")):
            continue
        gt = {}
        for line in open(f"{d_}/gt.tsv", encoding="utf-8").read().splitlines()[1:]:
            c = line.split("\t")
            if len(c) >= 5:
                gt[(c[0], int(c[1]))] = c[4]
        keys = []
        for line in open(f"{d_}/sub.mgf", encoding="utf-8"):
            if line.startswith("TITLE="):
                st, sc = line.strip()[6:].rsplit(":scan:", 1)
                keys.append((st, int(sc)))
        raw = mf.load_mztab(mzt)
        rows, n = [], 0
        for i, k in enumerate(keys):
            g = gt.get(k)
            if not g:
                continue
            n += 1
            p, sc = raw.get(i, ("", 0.0))
            if p:
                rows.append((key(sc, p), mf.full(p) == mf.full(g)))
        d = sweep(rows, n)
        base = 100 * sum(h for _, h in rows) / n
        print(f"{acc:<22}{n:>7,}{base:>11.1f}%{fmt(d,99.0):>20}{fmt(d,95.0):>20}{fmt(d,90.0):>20}")


if __name__ == "__main__":
    main()
