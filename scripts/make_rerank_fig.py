# -*- coding: utf-8 -*-
"""make_rerank_fig.py — 재순위 전용 그림 (신설 Figure 3).

왜 따로인가: 재순위는 "어휘와 학습이 있어야 하는가"(Figure 2)와 다른 질문이다.
이전 판본은 knapsack 을 Figure 2 의 막대 하나로만 두었는데, 정작 본문에는 그것만
다루는 절이 있고 그 절에 그림이 없었다. 두 질문을 그림으로도 분리한다.

세 팔은 후보를 **버리지 않고** 재정렬만 하므로 커버리지가 같다. 그래서 스펙트럼 단위
비교가 성립한다 — 패널 c 가 그 점을 눈으로 보인다.

  A  재순위 없음
  B  전구체 질량 (원고가 보고하는 설정)
  C  질량 + 참조 프로테옴의 트립신 공간

★ 2026-09-14 갱신 — arm C 를 확정 설정으로 교체했다.
  옛 arm C(2026-09-03)는 **DB 를 질량보다 먼저** 보는 경성 분할이었고 인덱스도 v1(시작 M 제거 누락)
  이었다. 새 arm C 는 **질량이 무조건 먼저, 그다음 DB 보너스 1.00**, 인덱스 v2 다. 외부 전체는
  거의 같지만(71.7 → 71.5%) DB 밖 정답 손실이 15건 → 4건으로 줄어든다(FACTS §15-6).

데이터: A·B = data/eval/dbrank, dbrank_ext (2026-09-03)
       C   = data/eval/rerank_verify (2026-09-11/14, `scripts/rerank_final.sh`)
사용: python scripts/make_rerank_fig.py
"""
from __future__ import annotations

import importlib.util
import io
from decimal import Decimal, ROUND_HALF_UP
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MASS_DB_DIR", "data/database")

_s = importlib.util.spec_from_file_location("mf_", os.path.join(BASE, "scripts", "make_figures.py"))
mf = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mf)
_d = importlib.util.spec_from_file_location("ds_", os.path.join(BASE, "scripts", "dbrank_score.py"))
ds = importlib.util.module_from_spec(_d)
_d.loader.exec_module(ds)

C_A, C_B, C_C = "#8d99a6", "#2f6f9f", "#4a7a3a"
C_TMT, C_LF, C_GREY = mf.C_TMT, mf.C_LF, mf.C_GREY
NL = chr(10)
ARMS = [("A", "no re-ranking", C_A), ("B", "precursor mass", C_B),
        ("C", "mass + tryptic search space", C_C)]


def arm_path(arm, acc=None):
    """arm C 만 확정 설정 산출물(rerank_verify)에서 읽는다."""
    if arm == "C":
        return f"data/eval/rerank_verify/{acc or 'internal'}.mztab"
    if acc is None:
        return f"data/eval/dbrank/arm{arm}.mztab"
    return f"data/eval/dbrank_ext/{acc}_{arm}.mztab"


def score_ext(acc, arm):
    """dbrank_score.score 와 같은 규칙(키 = (stem, scan))으로 임의 경로를 채점."""
    mz = arm_path(arm, acc)
    mgf = f"{ds.EXT}/{acc}/sub.mgf"
    if not (os.path.exists(mz) and os.path.exists(mgf)):
        return None
    gt = {}
    for line in io.open(f"{ds.EXT}/{acc}/gt.tsv", encoding="utf-8").read().splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 5:
            gt[(c[0], int(c[1]))] = c[4]
    raw = mf.load_mztab(mz)
    ok = n = cov = 0
    for i, k in enumerate(ds.keys_of(mgf)):
        g = gt.get(k)
        if not g:
            continue
        p = raw.get(i, raw.get(k[1], ("", 0)))[0]
        n += 1
        cov += bool(p)
        ok += bool(p) and mf.full(p) == mf.full(g)
    return ok, n, cov


def internal():
    out = {}
    for a, _, _ in ARMS:
        pred = mf.load_mztab(arm_path(a))
        st = {}
        for cls in ("tmt", "naked"):
            ok = n = cov = 0
            for idx, (c, seq) in mf.GT.items():
                if c != cls:
                    continue
                p, _sc = pred.get(idx, ("", 0.0))
                n += 1
                cov += bool(p)
                ok += bool(p) and mf.full(p) == mf.full(seq)
            st[cls] = (100 * ok / n, 100 * cov / n)
        out[a] = (st, pred)
    return out


def main():
    inte = internal()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6),
                             gridspec_kw={"width_ratios": [1.0, 1.55, 1.0]})
    fig.subplots_adjust(wspace=0.38, bottom=0.26)
    ax1, ax2, ax3 = axes

    # a — 내부 검증셋, 클래스별
    w = 0.26
    for j, (a, nm, c) in enumerate(ARMS):
        st = inte[a][0]
        vals = [st["tmt"][0], st["naked"][0]]
        xs = [i + (j - 1) * w for i in range(2)]
        ax1.bar(xs, vals, w, color=c, label=nm)
        for x_, v in zip(xs, vals):
            ax1.text(x_, v + 1.6, f"{v:.1f}", ha="center", fontsize=4.6)
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(["TMT", "label-free"], fontsize=8)
    ax1.set_ylim(0, 150)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylabel("Peptidoform precision (%)")
    # 2026-11-11: Fig. 3 F-H 로 합쳐졌다. 제목은 범례로.
    ax1.text(-0.14, 1.06, "F", transform=ax1.transAxes, fontsize=13, weight="bold", va="bottom")
    ax1.legend(loc="upper left", fontsize=5.6, frameon=False, ncol=1)

    # b — 외부 6종. 원고가 보고하는 것이 arm B 다.
    rows = []
    for acc, desc in ds.SETS:
        if acc in ("PXD054497", "PXD067164"):   # 2026-11-10: 원고는 공개 데이터 4건만 쓴다
            continue
        r = {x: score_ext(acc, x) for x in "ABC"}
        if any(v is None for v in r.values()):
            continue
        rows.append((acc, desc, [100 * r[x][0] / r[x][1] for x in "ABC"], r["B"][1]))
    # 2026-11-11: Fig. 3A 와 같은 순서(본문 서술 순서).
    ORDER = ["PXD054085", "PXD072211", "PXD063976", "PXD065357"]
    rows.sort(key=lambda r: ORDER.index(r[0]))
    xs = range(len(rows))
    for j, (a, nm, c) in enumerate(ARMS):
        v = [r[2][j] for r in rows]
        ax2.bar([i + (j - 1) * w for i in xs], v, w, color=c, label=nm)
    # 2026-11-12: 증가폭(+x) 대신 막대마다 precision 을 적는다(사용자 지시). 끝이 5 인 값은 사사오입.
    _q = lambda v: Decimal(repr(round(v, 6))).quantize(Decimal("0.1"), ROUND_HALF_UP)
    for i, r in enumerate(rows):
        for j in range(3):
            ax2.text(i + (j - 1) * w, r[2][j] + 1.5, f"{_q(r[2][j])}", ha="center",
                     fontsize=6.0, rotation=90, va="bottom")   # 세로로 — 이웃 막대 숫자와 겹치지 않게
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([r[0] + NL + r[1].split(",")[0] for r in rows],
                        fontsize=6.4, rotation=28, ha="right", rotation_mode="anchor")
    ax2.set_ylim(0, 172)   # 세로 숫자가 범례에 닿지 않게 (2026-11-18)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylabel("Peptidoform precision (%)")
    # 2026-11-11: Fig. 3 F-H 로 합쳐졌다. 제목은 범례로.
    ax2.text(-0.17, 1.04, "F", transform=ax2.transAxes, fontsize=11, weight="bold", va="bottom")
    ax2.legend(loc="upper left", fontsize=6.8, frameon=False)

    # c — 커버리지. 세 팔이 같다는 것이 이 규칙의 성질이다.
    for j, (a, nm, c) in enumerate(ARMS):
        st = inte[a][0]
        vals = [st["tmt"][1], st["naked"][1]]
        xs2 = [i + (j - 1) * w for i in range(2)]
        ax3.bar(xs2, vals, w, color=c, label=nm)
        if j == 1:   # 세 방식의 값이 같아 묶음마다 한 번만 적는다
            for x_, v in zip(xs2, vals):
                ax3.text(x_, v + 2.0, f"{v:.1f}", ha="center", fontsize=5.0)
    ax3.set_xticks(range(2))
    ax3.set_xticklabels(["TMT", "label-free"], fontsize=8)
    ax3.set_ylim(0, 150)
    ax3.set_yticks([0, 20, 40, 60, 80, 100])
    ax3.set_ylabel("Spectra given a peptide (%)")
    # 2026-11-11: Fig. 3 F-H 로 합쳐졌다. 제목은 범례로.
    ax3.text(-0.14, 1.06, "H", transform=ax3.transAxes, fontsize=13, weight="bold", va="bottom")
    ax3.text(0.5, 128, "no candidate is removed, so the same" + NL
             + "spectra are scored in all three modes",
             ha="center", va="bottom", fontsize=4.8, color=C_GREY)

    mf.suptitle(fig, "Figure 3 - Re-ranking the beam against the precursor mass "
                     "and the tryptic search space", y=1.04, fontsize=10.5, weight="bold")
    # 2026-11-11: test set 패널(a)과 예측 비율 패널(c)은 Fig. 2B 와 본문으로 옮겼다. 외부 데이터(b)만 Fig. 3F 로 남긴다.
    fig.delaxes(ax1)
    fig.delaxes(ax3)
    # 2026-11-18 (C26): 인쇄 크기로 — F 한 칸을 3.9 in 폭에 그린다 (G 2.6 in 와 합쳐 6.5 in)
    fig.set_size_inches(3.9, 2.8)
    from matplotlib.gridspec import GridSpec
    ax2.set_subplotspec(GridSpec(1, 1, figure=fig)[0])   # 남은 한 칸이 그림 전체를 쓰게 (tight_layout 이 옛 3칸 격자를 따르지 않게)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.30, top=0.92)
    import print_fit   # Fig. 3 아랫줄 F: 인쇄 폭 4.6 in (G 1.9 in 와 합쳐 6.5 in), 글자 7 pt 이상 (2026-11-18, C26)
    print_fit.save(fig, os.path.join(BASE, "figures"), "fig2b_rerank", 3.9)
    plt.close(fig)
    print("figures/fig2b_rerank.png / .pdf")
    tot = sum(r[3] for r in rows)
    for j, (a, nm, _c) in enumerate(ARMS):
        wavg = sum(r[3] * r[2][j] for r in rows) / tot
        print(f"  arm{a} {nm:<22} 외부 가중평균 {wavg:5.1f}%  (n={tot:,})")


if __name__ == "__main__":
    main()
