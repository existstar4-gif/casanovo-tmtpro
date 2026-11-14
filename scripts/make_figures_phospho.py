# -*- coding: utf-8 -*-
"""make_figures_phospho.py — 인산화 스터디 비교(Fig. 7)와 '우리만' 검증(Fig. 8)을 결과 파일에서 직접 그린다 (2026-11-03).

make_figures.py 의 원칙을 따른다: 요약 숫자를 옮겨 적지 않고 로그·JSON 을 읽어 그린다. 그림 안에 해석 문장은 넣지 않는다.

  F7  a 인산화 펩타이드 동정 수  b 정량 가능 수  c 검색 엔진 수확 대비 상대 이득 (7개 스터디)
  F8  a entrapment 오류율  b Sage 일치  c RT 일치  d 같은 인산화 자리 정량 상관

    python scripts/make_figures_phospho.py
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
C_TMT, C_LF, C_FP, C_GREY = "#2f6f9f", "#d98d2b", "#c0392b", "#495057"
C_GREEN, C_PALE = "#4a7a3a", "#d5dae0"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(BASE, "out", "quant")
V = os.path.join(Q, "validation")
OUT = os.path.join(BASE, "figures")
# 2026-11-10: 공개 원고에는 시료·유전형·조직 정보를 싣지 않는다(사용자 결정). 내부 이름 → Study 번호.
STUDIES = [("study1", "s1", "Study 1"), ("study2", "s2", "Study 2"), ("study3", "s3", "Study 3")]
DISPLAY = {"Study 1": "Study 1", "Study 2": "Study 2", "Study 3": "Study 3",
           "Study 4": "Study 4", "Study 5": "Study 5", "Study 6": "Study 6", "Study 7": "Study 7"}
METHODS = [("SEQUEST HT", C_PALE), ("CHIMERYS", C_GREY), ("FragPipe", C_LF), ("Casanovo-TMT", C_TMT)]
GROUPS = [("ours_conf", "unique")]   # 2026-11-13: one group, phosphopeptide-group threshold only


def scale_text(fig, k):
    """그림 안 모든 글자를 k 배로 (Fig. 3 과 같은 방식, 2026-11-13)."""
    import matplotlib.text as _mt
    for _t in fig.findobj(_mt.Text):
        _t.set_fontsize(_t.get_fontsize() * k)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def num(s):
    return int(s.replace(",", ""))


def split_counts(tag):
    t = open(os.path.join(Q, f"split_phospho_pfdr_lab_{tag}.log"), encoding="utf-8").read()
    pep = t.split("── 단백질")[0]
    out = {}
    for key, lab in (("Sequest HT (PD)", "SEQUEST HT"), ("CHIMERYS (PD)", "CHIMERYS"), ("우리 모델", "Casanovo-TMT")):
        m = re.search(re.escape(key) + r"\s+([\d,]+)", pep)
        if m:
            out[lab] = num(m.group(1))
    return out


def quant_counts(tag):
    t = open(os.path.join(Q, f"quant_share_rule_pfdr_lab_{tag}.log"), encoding="utf-8").read()
    out = {}
    for key, lab in (("Sequest HT", "SEQUEST HT"), ("CHIMERYS", "CHIMERYS"), ("Ours", "Casanovo-TMT")):
        m = re.search(r"^" + key + r"\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", t, re.M)
        out[lab] = num(m.group(3))
    return out



def panel(ax, letter, x=-0.16):
    """2026-11-13: 패널 제목 대신 굵은 대문자만 (Fig. 2·3 과 같은 양식, 제목은 범례로)."""
    ax.text(x, 1.04, letter, transform=ax.transAxes, fontsize=13, weight="bold", va="bottom")


def bars(ax, data, ylabel):
    w = 0.2
    x = np.arange(len(STUDIES))
    for i, (m, c) in enumerate(METHODS):
        v = [data[s][m] for s, _, _ in STUDIES]
        ax.bar(x + (i - 1.5) * w, v, w * 0.92, color=c, label=m, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([n for _, _, n in STUDIES])
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def fig7():
    ident, quant = {}, {}
    for tag, fp, _ in STUDIES:
        f = json.load(open(os.path.join(BASE, "out", "fragpipe_heldout", f"{fp}.json"), encoding="utf-8"))
        # 2026-11-13: FragPipe 는 자기 기본 출력(Philosopher 1%)을 쓴다 — PD 엔진처럼 각자의 기본값으로 비교(사용자 지시).
        ident[tag] = split_counts(tag); ident[tag]["FragPipe"] = f["A_default"]["phospho"]
        quant[tag] = quant_counts(tag); quant[tag]["FragPipe"] = f["A_default"]["quantifiable"]
    # 2026-11-13: 글자를 Fig. 3 과 같은 1.6 배로 키우려고 2x2 배치
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.6)); axs = axs.ravel()
    fig.subplots_adjust(hspace=0.45, wspace=0.45)
    bars(axs[0], ident, "Phosphopeptides identified")
    panel(axs[0], "A")
    axs[0].set_ylim(0, 21000)   # 2026-11-12: 범례가 Study 3 막대와 겹쳐 위 여백을 늘렸다
    axs[0].legend(ncol=2, loc="upper left")
    bars(axs[1], quant, "Quantifiable phosphopeptides")
    panel(axs[1], "B")
    # c (2026-11-12 신설, 사용자 요청): 네 방법의 겹침. Fig. 4A 와 같은 규칙의 집합
    #   (scripts/quant/overlap4_phospho.py, 집합 크기가 a 막대와 같음을 확인).
    #   합집합을 네 칸으로 나눈다: 넷 모두 / Casanovo-TMT + 일부 엔진 / Casanovo-TMT 만 / 엔진만.
    ax = axs[2]
    CATS = [("found by all four", C_TMT), ("Casanovo-TMT and some engines", "#8fb3d1"),
            ("unique to Casanovo-TMT", C_GREEN), ("unique to engines", C_PALE)]
    for i, (tag, _, name) in enumerate(STUDIES):
        d = json.load(open(os.path.join(V, f"overlap4_{tag}.json"), encoding="utf-8"))
        e, u = d["exact"], d["union"]
        all4 = e["SEQUEST HT & CHIMERYS & FragPipe & Casanovo-TMT"]
        only = e["Casanovo-TMT"]
        some = sum(v for k, v in e.items() if "Casanovo-TMT" in k) - all4 - only
        eng = sum(v for k, v in e.items() if "Casanovo-TMT" not in k)
        bottom = 0
        for (lab, col), v in zip(CATS, (all4, some, only, eng)):
            pct = 100 * v / u
            ax.bar(i, pct, 0.6, bottom=bottom, color=col, edgecolor="white", linewidth=0.5,
                   label=lab if i == 0 else None)
            ax.text(i, bottom + pct / 2, f"{v:,}", ha="center", va="center", fontsize=6.5,
                    color="white" if col in (C_TMT, C_GREEN) else "black")
            bottom += pct
    ax.set_xticks(range(len(STUDIES))); ax.set_xticklabels([n for _, _, n in STUDIES])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of all phosphopeptides (%)")
    panel(ax, "C", x=-0.30)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=1, fontsize=6.5)
    ax.grid(axis="x", visible=False)

    # d: 7개 스터디의 Sequest 수확 대비 상대 이득
    h = json.load(open(os.path.join(V, "harvest_gain.json"), encoding="utf-8"))
    ax = axs[3]
    for r in h["studies"]:
        held = r["held_out"]
        ax.scatter(r["sequest"], 100 * (r["ours"] / r["sequest"] - 1), s=36, zorder=3,
                   color=C_TMT if held else "white", edgecolor=C_TMT, linewidth=1.4)
        off = {"Study 4": (-4, -12), "Study 1": (6, 2), "Study 7": (-14, 6)}.get(r["name"], (5, 3))
        ax.annotate(DISPLAY[r["name"]], (r["sequest"], 100 * (r["ours"] / r["sequest"] - 1)), xytext=off,
                    textcoords="offset points", fontsize=7, color=C_GREY)
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Phosphopeptides identified by SEQUEST HT")
    ax.set_ylabel("More than SEQUEST HT (%)")
    panel(ax, "D", x=-0.30)
    ax.scatter([], [], color=C_TMT, label="not in corpus"); ax.scatter([], [], color="white", edgecolor=C_TMT, label="in corpus")
    # 2026-11-12: 범례와 Spearman 표기가 점과 섞였다 — 둘 다 그래프 밖 오른쪽으로 뺐다(사용자 지시).
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.text(1.04, 0.70, f"Spearman ρ = {h['rho']:.2f}" + chr(10) + f"n = {len(h['studies'])}", transform=ax.transAxes,
            fontsize=7.5, va="top")
    fig.tight_layout()
    scale_text(fig, 1.6)
    save(fig, "fig7_phospho_compare")


def fig8(fig=None, axs=None):
    """axs 를 주면 그 네 축에 A–D 만 그린다(Fig. 5 3열 합본, make_figures.fig4 에서 부른다; 2026-11-13)."""
    own = fig is None
    if own:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9.6)); axs = axs.ravel()   # 2026-11-13: 2x2, 글자 1.6 배
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # a entrapment: 애기장대(결합 추정) 와 맞바꾸기(짝 추정)
    ax = axs[0]
    ef = {t: json.load(open(os.path.join(V, f"entrapment_formal_{t}.json"), encoding="utf-8")) for t, _, _ in STUDIES}
    # 2026-11-13: 다른 종(애기장대) entrapment 결합 추정만 그린다(사용자 지시). 점선 = 1%, 5%.
    cats = [("all_phospho", "all" + chr(10) + "phospho")] + GROUPS
    x = np.arange(len(cats)); w = 0.25
    for j, (t, _, name) in enumerate(STUDIES):
        fo = [100 * ef[t][c]["arabidopsis"]["fdp_combined"] if ef[t][c]["arabidopsis"]["N_T"] else np.nan for c, _ in cats]
        ax.bar(x + (j - 1) * w, fo, w * 0.92, color=C_TMT, alpha=[0.45, 0.7, 1.0][j], edgecolor="white", lw=0.4)
    for yv, ls in ((5, "--"),):   # 2026-11-13: 5% 선만 (사용자 지시)
        ax.axhline(yv, color=C_GREY, lw=0.9, ls=ls)
        ax.text(1.0, yv, f" {yv}%", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7, color=C_GREY)
    ax.set_xticks(x); ax.set_xticklabels([c for _, c in cats], fontsize=7)
    ax.set_ylabel("Estimated FDP (%)")
    panel(ax, "A", x=-0.2); ax.grid(axis="x", visible=False)
    # b Sage 확신 스캔에서 같은 서열 비율 (예측 스펙트럼 각도는 판별력이 없어(AUC 0.43–0.60) 그림에서 뺐다)
    ax = axs[1]
    sg = json.load(open(os.path.join(V, "sage_recheck.json"), encoding="utf-8"))
    cats = [("shared", "shared")] + GROUPS
    x = np.arange(len(cats)); w = 0.25
    for j, (t, _, name) in enumerate(STUDIES):
        v, n = [], []
        for c, _ in cats:
            d = sg[t].get(c, {})
            v.append(100 * d["conf_same_frac"] if d.get("conf") else np.nan); n.append(d.get("conf", 0))
        ax.bar(x + (j - 1) * w, v, w * 0.92, color=C_TMT, alpha=[0.45, 0.7, 1.0][j], edgecolor="white", lw=0.4)
        for xi, vi, ni in zip(x + (j - 1) * w, v, n):
            if ni:
                ax.text(xi, 2, str(ni), ha="center", va="bottom", fontsize=5.5, rotation=90, color="white")
    ax.set_ylim(0, 105)
    ax.set_xticks(x); ax.set_xticklabels([c for _, c in cats], fontsize=7)
    ax.set_ylabel("Same sequence as Sage (%)")
    panel(ax, "B", x=-0.2); ax.grid(axis="x", visible=False)
    # c RT 일치 (2026-11-13: 막대 대신 누적분포 — 세 스터디를 합치고, 오차는 공유 95분위로 나눠 1 이 경계)
    ax = axs[2]
    rel = collections.defaultdict(list)
    for t, _, _ in STUDIES:
        d = json.load(open(os.path.join(V, f"ours_only_validation_{t}.json"), encoding="utf-8"))["C_rt"]["_rel_err"]
        for g, v in d.items():
            rel[g] += v
    for g, lab, col, ls in (("shared", "shared", C_TMT, "-"), ("ours_conf", "unique", C_GREEN, "-"),
                            ("wrong", "decoy + foreign", C_GREY, "--")):
        v = np.sort(np.array(rel[g]))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=2.0, ls=ls, label=lab)
    ax.axvline(1, color=C_GREY, lw=0.8, ls=":")
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.0)
    ax.set_xlabel("RT error / shared 95th percentile")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="lower right", fontsize=7.5)
    panel(ax, "C", x=-0.2)
    # d 같은 자리 정량
    for ax, key, field, ylab, title in ((axs[3], "K_same_site_partner", "mean_r", "Reporter correlation", "D"),):
        cats = [("shared", "shared")] + GROUPS
        x = np.arange(len(cats)); w = 0.25
        for j, (t, _, name) in enumerate(STUDIES):
            d = json.load(open(os.path.join(V, f"ours_only_validation_{t}.json"), encoding="utf-8"))[key]
            v = [d.get(c, {}).get(field) if d.get(c, {}).get("n") else np.nan for c, _ in cats]
            v = [np.nan if a is None else a for a in v]
            ax.bar(x + (j - 1) * w, v, w * 0.92, color=C_TMT, alpha=[0.45, 0.7, 1.0][j], label=name, edgecolor="white", lw=0.4)
        ax.set_xticks(x); ax.set_xticklabels([c for _, c in cats], fontsize=7)
        ax.set_ylabel(ylab); panel(ax, title, x=-0.2); ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    hs = [Patch(color=C_TMT, alpha=a, label=n) for a, (_, _, n) in zip([0.45, 0.7, 1.0], STUDIES)]
    if not own:   # 합본: 스터디 진하기 범례를 A 안 왼쪽 위에 둔다
        axs[0].legend(handles=hs, loc="upper left", fontsize=7.5)
        return
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), title="shading: study", fontsize=7.5)
    fig.tight_layout()
    scale_text(fig, 1.6)
    save(fig, "fig8_ours_only_validation")


if __name__ == "__main__":
    which = sys.argv[1:] or ["7", "8"]
    if "7" in which:
        fig7()
    if "8" in which:
        fig8()
