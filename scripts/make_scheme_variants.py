# -*- coding: utf-8 -*-
"""make_scheme_variants.py — Figure 1 배치 시안 5종 (2026-11-11).

내용은 같고 배치만 다르다. 사용자 지시: 코퍼스는 "27,075,536 MS spectra / 2,146 runs" 만,
split-by-peptide 상자·pS/pT/pY 비율·mzTab 상자는 뺀다. 평가는 test set(test 1)만.
그린 뒤 글자가 상자 밖으로 나가는지 렌더러로 검사해 경고한다.

사용: python scripts/make_scheme_variants.py   → figures/fig1_variants/V1..V5.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "figures", "fig1_variants")
os.makedirs(OUT, exist_ok=True)

C_TMT, C_LF, C_FP, C_GREY = "#2f6f9f", "#d98d2b", "#c0392b", "#495057"
C_GREEN, C_PALE, C_INK = "#4a7a3a", "#d5dae0", "#1b2733"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300,
                     "savefig.bbox": "tight", "figure.dpi": 120})
NL = chr(10)

# ── 공통 내용 ───────────────────────────────────────────────────────────
CORPUS = "27,075,536 MS spectra" + NL + "2,146 runs"
SPLIT = (("training", "21,688,803", C_INK, "white"),
         ("validation", "2,717,217", C_PALE, C_INK),
         ("test", "2,669,516", C_GREEN, "white"))
TMT_RUNS = "TMTpro runs" + NL + "18,283,089 (86.9%)"
LF_RUNS = "label-free runs" + NL + "2,753,586 (13.1%)"
TAGGED = "tagged peptides 18,211,829" + NL + "phosphorylated 13.0%"
SPEC_IN = "MS2 spectrum" + NL + "and precursor m/z, charge"
ENC_DEC = "Casanovo 5.2.0" + NL + "encoder-decoder transformer"
VOCAB = "Vocabulary extended" + NL + "[TMTpro]-  K[TMTpro]  pS  pT  pY"
FINETUNE = "Fine-tuned on the training partition" + NL + "3,300,000 steps, batch 64"
BEAM = "Beam search" + NL + "ten candidates"
RERANK = "Precursor-mass re-ranking" + NL + "matches within 20 ppm first"
OUTPUT = "Peptidoform" + NL + "sequence and modifications"
EVAL = (("Test set, test partition" + NL + "3,000 TMTpro" + NL + "1,000 label-free",
         "TMTpro 84.7%" + NL + "label-free 61.2%"),
        ("4 public datasets" + NL + "other laboratories" + NL + "4,000 spectra each",
         "87.4, 83.3, 79.5" + NL + "and 69.2%"),
        ("Studies 1-3" + NL + "phosphoproteomes" + NL + "not in training",
         "19.7-44.8% more" + NL + "phosphopeptides" + NL + "than SEQUEST HT"))
CONTROLS = ("Controls:  vocabulary extended but untrained 0.3%  ·  "
            "re-ranking disabled, -1.5 to -3.6 points on the public datasets")

_CHECK = []


def box(ax, x, y, w, h, text, fc, ec=None, fs=7.0, tc="white", weight="normal", r=0.018):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                                linewidth=1.0, facecolor=fc, edgecolor=ec or fc, zorder=2))
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, zorder=3, weight=weight, linespacing=1.35)
    _CHECK.append((ax, t, (x, y, w, h)))


def group(ax, x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018",
                                linewidth=1.1, facecolor="none", edgecolor=color,
                                linestyle=(0, (4, 2.6)), zorder=1))
    ax.text(x + 0.010, y + h - 0.012, label, ha="left", va="top", fontsize=8.0,
            color=color, weight="bold", zorder=3)


def label(ax, x, y, s, color=C_INK, fs=11):
    ax.text(x, y, s, ha="left", va="top", fontsize=fs, weight="bold", color=color, zorder=5)


def arrow(ax, p1, p2, color=C_GREY, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
                                 color=color, zorder=4, connectionstyle="arc3,rad=%s" % rad))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for ax, t, (x, y, w, h) in _CHECK:
        if ax.figure is not fig:
            continue
        bb = t.get_window_extent(r)
        x0, y0 = ax.transData.transform((x, y))
        x1, y1 = ax.transData.transform((x + w, y + h))
        if bb.x0 < x0 - 1 or bb.x1 > x1 + 1 or bb.y0 < y0 - 1 or bb.y1 > y1 + 1:
            bad += 1
            print(f"  ⚠ {name}: 넘침 {t.get_text().splitlines()[0]!r}")
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"{name}: 넘침 {bad}")


def split_row(ax, x, y, w, h, fs=6.6):
    g = 0.012
    pw = (w - 2 * g) / 3
    for i, (lab, n, fc, tc) in enumerate(SPLIT):
        box(ax, x + i * (pw + g), y, pw, h, lab + NL + n, fc, tc=tc, fs=fs)


# ── V1: 왼쪽→오른쪽 한 줄 파이프라인 (4열) ─────────────────────────────
def v1():
    fig, ax = canvas(14.0, 5.0)
    cols = [(0.005, "Corpus", C_GREY), (0.255, "Fine-tuning", C_TMT),
            (0.505, "Decoding", C_FP), (0.755, "Evaluation", C_GREEN)]
    W = 0.235
    for x, lab, c in cols:
        group(ax, x, 0.03, W, 0.94, lab, c)
    x = 0.017; bw = W - 0.024
    box(ax, x, 0.76, bw, 0.13, CORPUS, C_INK, fs=8.2, weight="bold")
    split_row(ax, x, 0.58, bw, 0.13)
    box(ax, x, 0.39, bw / 2 - 0.004, 0.14, TMT_RUNS, C_TMT, fs=6.6)
    box(ax, x + bw / 2 + 0.004, 0.39, bw / 2 - 0.004, 0.14, LF_RUNS, C_LF, fs=6.6)
    box(ax, x, 0.20, bw, 0.14, TAGGED, C_TMT, fs=7.0)
    ax.text(x + bw / 2, 0.36, "composition of the training partition", ha="center",
            va="center", fontsize=6.2, style="italic", color=C_GREY)
    x = 0.267
    box(ax, x, 0.76, bw, 0.13, SPEC_IN, C_INK, fs=7.4)
    box(ax, x, 0.55, bw, 0.15, ENC_DEC, C_TMT, fs=7.6)
    box(ax, x, 0.34, bw, 0.15, VOCAB, C_GREEN, fs=7.0)
    box(ax, x, 0.13, bw, 0.15, FINETUNE, "white", ec=C_GREEN, tc=C_INK, fs=6.8)
    for y in (0.76, 0.55):
        arrow(ax, (x + bw / 2, y), (x + bw / 2, y - 0.045))
    x = 0.517
    box(ax, x, 0.70, bw, 0.17, BEAM, C_FP, fs=7.6)
    box(ax, x, 0.44, bw, 0.17, RERANK, C_FP, fs=7.0)
    box(ax, x, 0.18, bw, 0.17, OUTPUT, C_INK, fs=7.4)
    for y in (0.70, 0.44):
        arrow(ax, (x + bw / 2, y), (x + bw / 2, y - 0.075))
    x = 0.767
    for i, (top, res) in enumerate(EVAL):
        y = 0.66 - i * 0.29
        box(ax, x, y, bw * 0.55, 0.24, top, C_GREEN, fs=5.9)
        box(ax, x + bw * 0.57, y, bw * 0.43, 0.24, res, C_PALE, tc=C_INK, fs=6.4)
    for a, b in ((0.241, 0.259), (0.491, 0.509), (0.741, 0.759)):
        arrow(ax, (a, 0.5), (b, 0.5), lw=1.8)
    ax.text(0.5, -0.02, CONTROLS, ha="center", va="top", fontsize=6.6, color=C_INK)
    save(fig, "V1_pipeline_row")


# ── V2: 위 3패널 + 아래 평가 띠 — 2026-11-11 최종본(Figure 1) ──────────────
def v2():
    fig, ax = canvas(13.6, 6.0)
    # 코퍼스
    group(ax, 0.010, 0.40, 0.262, 0.585, "In-house corpus", C_GREY)
    box(ax, 0.024, 0.83, 0.234, 0.09, CORPUS, C_INK, fs=8.2, weight="bold")
    split_row(ax, 0.024, 0.715, 0.234, 0.085)
    box(ax, 0.024, 0.575, 0.110, 0.11, "training," + NL + TMT_RUNS, C_TMT, fs=6.0)
    box(ax, 0.148, 0.575, 0.110, 0.11, "training," + NL + LF_RUNS, C_LF, fs=6.0)
    box(ax, 0.024, 0.435, 0.234, 0.11,
        "training, tagged peptides 18,211,829" + NL + "phosphorylated 13.0%", C_TMT, fs=6.6)
    # 모델
    group(ax, 0.300, 0.40, 0.330, 0.585, "Model and fine-tuning", C_TMT)
    box(ax, 0.316, 0.83, 0.298, 0.09, SPEC_IN, C_INK, fs=7.4)
    box(ax, 0.316, 0.685, 0.298, 0.09, ENC_DEC, C_TMT, fs=7.4)
    box(ax, 0.316, 0.435, 0.141, 0.15,
        "Vocabulary extended" + NL + "[TMTpro]-  K[TMTpro]" + NL + "pS  pT  pY", C_GREEN, fs=6.5)
    box(ax, 0.473, 0.435, 0.141, 0.15,
        "Fine-tuned on the" + NL + "training partition" + NL + "3,300,000 steps" + NL + "batch 64",
        "white", ec=C_GREEN, tc=C_INK, fs=6.3)
    arrow(ax, (0.465, 0.828), (0.465, 0.778))
    arrow(ax, (0.3865, 0.587), (0.3865, 0.683), color=C_GREEN)
    arrow(ax, (0.5435, 0.587), (0.5435, 0.683), color=C_GREEN)
    # 디코딩
    group(ax, 0.656, 0.40, 0.334, 0.585, "Decoding", C_FP)
    box(ax, 0.672, 0.81, 0.302, 0.10, BEAM, C_FP, fs=7.6)
    box(ax, 0.672, 0.635, 0.302, 0.10, RERANK, C_FP, fs=7.2)
    box(ax, 0.672, 0.46, 0.302, 0.10, OUTPUT, C_INK, fs=7.4)
    arrow(ax, (0.823, 0.808), (0.823, 0.737))
    arrow(ax, (0.823, 0.633), (0.823, 0.562))
    # 칸 사이 흐름
    arrow(ax, (0.274, 0.49), (0.298, 0.49), lw=1.8)
    arrow(ax, (0.632, 0.73), (0.654, 0.73), lw=1.8)
    arrow(ax, (0.823, 0.397), (0.823, 0.373), color=C_GREEN, lw=1.8)
    # 평가
    group(ax, 0.010, 0.015, 0.980, 0.355, "Evaluation", C_GREEN)
    U, GU = 0.303, 0.0195
    for i, (top, res) in enumerate(EVAL):
        x = 0.026 + i * (U + GU)
        box(ax, x, 0.145, 0.168, 0.15, top, C_GREEN, fs=6.3)
        box(ax, x + 0.187, 0.145, 0.116, 0.15, res, C_PALE, tc=C_INK, fs=6.5)
        arrow(ax, (x + 0.170, 0.22), (x + 0.185, 0.22))
    box(ax, 0.026, 0.045, 0.948, 0.065, CONTROLS, "white", ec=C_GREEN, tc=C_INK, fs=6.8)
    save(fig, "V2_three_panels_eval_band")


# ── V3: 학습 경로(위)와 추론 경로(아래)를 나눈 두 줄 ──────────────────
def v3():
    fig, ax = canvas(14.0, 6.0)
    ax.text(0.005, 0.975, "Training", fontsize=9, weight="bold", color=C_TMT, va="top")
    ax.text(0.005, 0.50, "Inference and evaluation", fontsize=9, weight="bold", color=C_FP, va="top")
    ax.add_patch(Rectangle((0.0, 0.535), 1.0, 0.002, color=C_PALE))
    box(ax, 0.010, 0.72, 0.17, 0.17, CORPUS, C_INK, fs=8.0, weight="bold")
    split_row(ax, 0.200, 0.79, 0.25, 0.10, fs=6.4)
    box(ax, 0.200, 0.63, 0.122, 0.12, TMT_RUNS, C_TMT, fs=6.0)
    box(ax, 0.328, 0.63, 0.122, 0.12, LF_RUNS, C_LF, fs=6.0)
    ax.text(0.325, 0.605, "training partition", ha="center", fontsize=6.0, style="italic",
            color=C_GREY, va="center")
    box(ax, 0.480, 0.72, 0.17, 0.17, VOCAB, C_GREEN, fs=6.8)
    box(ax, 0.680, 0.72, 0.17, 0.17, FINETUNE, "white", ec=C_GREEN, tc=C_INK, fs=6.2)
    box(ax, 0.870, 0.72, 0.125, 0.17, "fine-tuned" + NL + "model", C_TMT, fs=8.0, weight="bold")
    for a, b in ((0.182, 0.198), (0.452, 0.478), (0.652, 0.678), (0.852, 0.868)):
        arrow(ax, (a, 0.805), (b, 0.805), lw=1.6)
    box(ax, 0.010, 0.30, 0.15, 0.14, SPEC_IN, C_INK, fs=6.6)
    box(ax, 0.180, 0.30, 0.15, 0.14, "fine-tuned" + NL + "model", C_TMT, fs=7.6)
    box(ax, 0.350, 0.30, 0.15, 0.14, BEAM, C_FP, fs=7.0)
    box(ax, 0.520, 0.30, 0.17, 0.14, RERANK, C_FP, fs=6.4)
    box(ax, 0.710, 0.30, 0.15, 0.14, OUTPUT, C_INK, fs=6.4)
    for a, b in ((0.162, 0.178), (0.332, 0.348), (0.502, 0.518), (0.692, 0.708)):
        arrow(ax, (a, 0.37), (b, 0.37), lw=1.6)
    arrow(ax, (0.932, 0.72), (0.27, 0.442), rad=-0.05, color=C_TMT)
    for i, (top, res) in enumerate(EVAL):
        x = 0.010 + i * 0.330
        box(ax, x, 0.07, 0.19, 0.17, top, C_GREEN, fs=6.2)
        box(ax, x + 0.195, 0.07, 0.12, 0.17, res, C_PALE, tc=C_INK, fs=6.4)
    arrow(ax, (0.785, 0.30), (0.785, 0.245), color=C_GREEN)
    ax.text(0.5, 0.02, CONTROLS, ha="center", va="top", fontsize=6.6, color=C_INK)
    save(fig, "V3_training_vs_inference")


# ── V4: 세로 한 단 (단일 칼럼 저널용) ───────────────────────────────────
def v4():
    fig, ax = canvas(6.2, 10.0)
    secs = [(0.775, 0.215, "Corpus", C_GREY), (0.535, 0.225, "Model and fine-tuning", C_TMT),
            (0.330, 0.190, "Decoding", C_FP), (0.015, 0.300, "Evaluation", C_GREEN)]
    for y, h, lab, c in secs:
        group(ax, 0.01, y, 0.98, h, lab, c)
    box(ax, 0.04, 0.89, 0.42, 0.065, CORPUS, C_INK, fs=8.0, weight="bold")
    split_row(ax, 0.50, 0.89, 0.46, 0.065, fs=6.4)
    box(ax, 0.04, 0.795, 0.28, 0.07, TMT_RUNS, C_TMT, fs=6.6)
    box(ax, 0.34, 0.795, 0.28, 0.07, LF_RUNS, C_LF, fs=6.6)
    box(ax, 0.64, 0.795, 0.32, 0.07, TAGGED, C_TMT, fs=6.0)
    box(ax, 0.04, 0.665, 0.44, 0.065, SPEC_IN, C_INK, fs=7.0)
    box(ax, 0.52, 0.665, 0.44, 0.065, ENC_DEC, C_TMT, fs=7.0)
    box(ax, 0.04, 0.555, 0.44, 0.075, VOCAB, C_GREEN, fs=6.8)
    box(ax, 0.52, 0.555, 0.44, 0.075, FINETUNE, "white", ec=C_GREEN, tc=C_INK, fs=6.2)
    arrow(ax, (0.48, 0.6975), (0.518, 0.6975))
    box(ax, 0.04, 0.37, 0.27, 0.10, BEAM, C_FP, fs=7.0)
    box(ax, 0.36, 0.37, 0.30, 0.10, RERANK, C_FP, fs=6.2)
    box(ax, 0.71, 0.37, 0.25, 0.10, "Peptidoform" + NL + "sequence and" + NL + "modifications", C_INK, fs=6.4)
    arrow(ax, (0.312, 0.42), (0.358, 0.42))
    arrow(ax, (0.662, 0.42), (0.708, 0.42))
    for i, (top, res) in enumerate(EVAL):
        y = 0.215 - i * 0.085
        box(ax, 0.04, y, 0.58, 0.075, top.replace(NL, ", ", 1), C_GREEN, fs=6.2)
        box(ax, 0.64, y, 0.32, 0.075, res, C_PALE, tc=C_INK, fs=6.4)
    for y in (0.775, 0.535, 0.330):
        arrow(ax, (0.5, y), (0.5, y - 0.025), lw=1.8)
    save(fig, "V4_vertical")


# ── V5: A/B/C 패널 — 코퍼스는 막대 그래픽, 결과는 큰 숫자 ───────────────
def v5():
    fig, ax = canvas(13.6, 6.0)
    label(ax, 0.0, 1.0, "A")
    ax.text(0.03, 0.995, CORPUS.replace(NL, ", "), fontsize=9, weight="bold", color=C_INK, va="top")
    tot = 21688803 + 2717217 + 2669516
    x, y0, w = 0.03, 0.82, 0.40
    for lab, n, fc, tc in SPLIT:
        f = int(n.replace(",", "")) / tot * w
        ax.add_patch(Rectangle((x, y0), f - 0.002, 0.08, color=fc, zorder=2))
        ax.text(x + f / 2, y0 + 0.04, lab + NL + n, ha="center", va="center",
                fontsize=6.4, color=tc, zorder=3)
        x += f
    x = 0.03
    tr = 18283089 + 2753586
    for lab, n, fc in (("TMTpro runs", 18283089, C_TMT), ("label-free", 2753586, C_LF)):
        f = n / tr * (21036675 / tot * w)
        ax.add_patch(Rectangle((x, 0.70), f - 0.002, 0.07, color=fc, zorder=2))
        ax.text(x + f / 2, 0.735, f"{lab}" + NL + f"{n:,}", ha="center", va="center",
                fontsize=6.0, color="white", zorder=3)
        x += f
    ax.text(0.03, 0.685, "training partition: phosphorylated 13.0% of tagged-peptide spectra",
            fontsize=6.4, color=C_GREY, style="italic", va="top")
    label(ax, 0.47, 1.0, "B")
    seq = [("MS2 spectrum" + NL + "+ precursor", C_INK), ("fine-tuned" + NL + "Casanovo", C_TMT),
           ("beam search" + NL + "10 candidates", C_FP), ("precursor-mass" + NL + "re-ranking", C_FP),
           ("peptidoform", C_INK)]
    for i, (t, c) in enumerate(seq):
        xx = 0.50 + i * 0.100
        box(ax, xx, 0.78, 0.092, 0.15, t, c, fs=6.4)
        if i:
            arrow(ax, (xx - 0.008, 0.855), (xx, 0.855))
    box(ax, 0.50, 0.60, 0.192, 0.12, VOCAB, C_GREEN, fs=6.4)
    box(ax, 0.70, 0.60, 0.192, 0.12, FINETUNE, "white", ec=C_GREEN, tc=C_INK, fs=5.9)
    arrow(ax, (0.646, 0.72), (0.646, 0.778), color=C_GREEN)
    label(ax, 0.0, 0.52, "C")
    big = (("84.7%", "TMTpro, test set" + NL + "label-free 61.2%"),
           ("69.2-87.4%", "four public datasets" + NL + "other laboratories"),
           ("+19.7-44.8%", "phosphopeptides vs SEQUEST HT" + NL + "Studies 1-3"))
    for i, (num, sub) in enumerate(big):
        xx = 0.03 + i * 0.33
        ax.add_patch(FancyBboxPatch((xx, 0.13), 0.30, 0.33, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    facecolor="#eef2ee", edgecolor=C_GREEN, linewidth=1.0))
        ax.text(xx + 0.15, 0.34, num, ha="center", va="center", fontsize=20, weight="bold", color=C_GREEN)
        ax.text(xx + 0.15, 0.21, sub, ha="center", va="center", fontsize=7.2, color=C_INK, linespacing=1.35)
    ax.text(0.5, 0.07, CONTROLS, ha="center", va="top", fontsize=6.6, color=C_INK)
    save(fig, "V5_panels_ABC")


# ── 최종본 (2026-11-11): V2 배치를 인쇄 폭 7.1 in 로 다시 그림. 글자 6 pt 이상 ─────────
def final():
    fig, ax = canvas(7.1, 4.0)

    def b(x, y, w, h, t, fc, ec=None, fs=6.2, tc="white", weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012",
                                    linewidth=0.8, facecolor=fc, edgecolor=ec or fc, zorder=2))
        tt = ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=tc,
                     zorder=3, weight=weight, linespacing=1.3)
        _CHECK.append((ax, tt, (x, y, w, h)))

    def g(x, y, w, h, lab, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    linewidth=0.9, facecolor="none", edgecolor=c,
                                    linestyle=(0, (4, 2.4)), zorder=1))
        ax.text(x + 0.008, y + h - 0.010, lab, ha="left", va="top", fontsize=7.5,
                color=c, weight="bold", zorder=3)

    def ar(p1, p2, c=C_GREY, lw=1.0):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8, linewidth=lw,
                                     color=c, zorder=4))

    # 코퍼스
    X0, CW = 0.008, 0.300
    g(X0, 0.40, CW, 0.585, "In-house corpus", C_GREY)
    xi, wi = X0 + 0.015, CW - 0.030
    b(xi, 0.835, wi, 0.085, CORPUS, C_INK, fs=7.5, weight="bold")
    pw = (wi - 0.016) / 3
    for i, (lab, n, fc, tc) in enumerate(SPLIT):
        b(xi + i * (pw + 0.008), 0.725, pw, 0.085, lab + NL + n, fc, tc=tc, fs=6.0)
    ax.text(xi, 0.700, "training partition", ha="left", va="center", fontsize=6.0,
            weight="bold", color=C_GREY)
    hw = (wi - 0.010) / 2
    b(xi, 0.555, hw, 0.125, "TMTpro runs" + NL + "18,283,089" + NL + "spectra", C_TMT, fs=6.0)
    b(xi + hw + 0.010, 0.555, hw, 0.125, "label-free runs" + NL + "2,753,586" + NL + "spectra", C_LF, fs=6.0)
    b(xi, 0.430, wi, 0.100, "tagged peptides 18,211,829" + NL + "13.0% phosphorylated", C_TMT, fs=6.2)
    # 모델
    X1, MW = 0.333, 0.322
    g(X1, 0.40, MW, 0.585, "Model and fine-tuning", C_TMT)
    xi, wi = X1 + 0.015, MW - 0.030
    b(xi, 0.835, wi, 0.085, "MS2 spectrum" + NL + "precursor m/z and charge", C_INK, fs=6.2)
    b(xi, 0.695, wi, 0.090, "Casanovo 5.2.0" + NL + "encoder-decoder transformer", C_TMT, fs=6.4)
    hw = (wi - 0.012) / 2
    b(xi, 0.425, hw, 0.200, "Vocabulary" + NL + "extended" + NL + "[TMTpro]-  K[TMTpro]".replace("  ", NL) + NL + "pS  pT  pY",
      C_GREEN, fs=6.0)
    b(xi + hw + 0.012, 0.425, hw, 0.200, "Fine-tuned on" + NL + "training partition" + NL
      + "3,300,000 steps" + NL + "batch 64", "white", ec=C_GREEN, tc=C_INK, fs=6.0)
    ar((xi + wi / 2, 0.833), (xi + wi / 2, 0.787))
    ar((xi + hw / 2, 0.627), (xi + hw / 2, 0.693), C_GREEN)
    ar((xi + hw * 1.5 + 0.012, 0.627), (xi + hw * 1.5 + 0.012, 0.693), C_GREEN)
    # 디코딩
    X2, DW = 0.680, 0.312
    g(X2, 0.40, DW, 0.585, "Decoding", C_FP)
    xi, wi = X2 + 0.015, DW - 0.030
    b(xi, 0.815, wi, 0.095, "Beam search" + NL + "ten candidates", C_FP, fs=6.4)
    b(xi, 0.640, wi, 0.105, "Candidates matching" + NL + "precursor mass (20 ppm)" + NL + "ranked first",
      C_FP, fs=6.2)
    b(xi, 0.465, wi, 0.105, "Peptidoform" + NL + "sequence and modifications", C_INK, fs=6.4)
    cx = xi + wi / 2
    ar((cx, 0.813), (cx, 0.747))
    ar((cx, 0.638), (cx, 0.572))
    # 칸 사이
    ar((X0 + CW + 0.003, 0.49), (X1 - 0.003, 0.49), lw=1.3)
    ar((X1 + MW + 0.003, 0.74), (X2 - 0.003, 0.74), lw=1.3)
    ar((cx, 0.397), (cx, 0.373), C_GREEN, lw=1.3)
    # 평가
    g(0.008, 0.015, 0.984, 0.355, "Evaluation", C_GREEN)
    tiles = (("Test set" + NL + "3,000 TMTpro" + NL + "1,000 label-free",
              "peptidoform" + NL + "precision" + NL + "TMTpro 84.7%" + NL + "label-free 61.2%"),
             ("4 public datasets" + NL + "other laboratories" + NL + "4,000 spectra each",
              "peptidoform" + NL + "precision" + NL + "87.4, 83.3," + NL + "79.5 and 69.2%"),
             ("Studies 1-3" + NL + "phosphoproteomes" + NL + "not in training",
              "phosphopeptides" + NL + "19.7-44.8% more" + NL + "than SEQUEST HT"))
    U, GU, GW = 0.312, 0.012, 0.150
    for i, (t, r) in enumerate(tiles):
        x = 0.023 + i * (U + GU)
        b(x, 0.150, GW, 0.165, t, C_GREEN, fs=6.0)
        b(x + GW + 0.016, 0.150, U - GW - 0.016, 0.165, r, C_PALE, tc=C_INK, fs=6.0)
        ar((x + GW + 0.002, 0.2325), (x + GW + 0.014, 0.2325))
    b(0.023, 0.035, 0.954, 0.090,
      "Controls:  vocabulary extended but untrained, 0.3% on the test set" + NL
      + "re-ranking disabled, 1.5 to 3.6 points lower on the public datasets",
      "white", ec=C_GREEN, tc=C_INK, fs=6.0)
    save(fig, "FINAL_print")
    for ext in ("png", "pdf"):
        src = os.path.join(OUT, "FINAL_print." + ext)
        for dst in (os.path.join(BASE, "figures", "fig0_scheme." + ext),
                    os.path.join(BASE, "figures", "manuscript_v3", "Figure1." + ext)):
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                fo.write(fi.read())


# ── 최종본 v2 (2026-11-11): 같은 배치, 현대적 스타일 ───────────────────────
# 꽉 찬 색 상자 대신 흰 카드 + 색 띠, 옅은 배경 판, Arial, 숫자 강조, 스펙트럼·서열 예시.
def modern():
    plt.rcParams["font.family"] = "Arial"
    fig, ax = canvas(7.1, 4.1)
    INK, SUB, LINE, BG = "#1b2733", "#5b6670", "#d5dae0", "#f3f5f7"
    BLUE, ORANGE, GREEN, RED = "#2f6f9f", "#d98d2b", "#4a7a3a", "#b8403a"
    MONO = "Consolas"
    checks = []

    def panel(x, y, w, h, title, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.014",
                                    facecolor=BG, edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((x + 0.012, y + h - 0.040), 0.004, 0.022, color=c, zorder=1))
        ax.text(x + 0.022, y + h - 0.029, title, ha="left", va="center", fontsize=7.6,
                weight="bold", color=INK, zorder=2)

    def card(x, y, w, h, c=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.010",
                                    facecolor="white", edgecolor=LINE, linewidth=0.6, zorder=1))
        if c:
            ax.add_patch(FancyBboxPatch((x, y), 0.006, h, boxstyle="round,pad=0,rounding_size=0.003",
                                        facecolor=c, edgecolor="none", zorder=2))
        return (x, y, w, h)

    def t(x, y, s, fs=6.2, c=INK, w="normal", ha="left", va="center", fam=None, box=None):
        kw = {"family": fam} if fam else {}
        tt = ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va, zorder=3,
                     linespacing=1.3, **kw)
        if box:
            checks.append((tt, box))
        return tt

    def ar(p1, p2, c="#8a949e", lw=0.9):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7, linewidth=lw,
                                     color=c, zorder=4))

    TOP, H = 0.43, 0.555
    # ── 코퍼스 ──
    X0, W0 = 0.008, 0.300
    panel(X0, TOP, W0, H, "In-house corpus", INK)
    xi, wi = X0 + 0.022, W0 - 0.044
    pb = (X0, TOP, W0, H)
    t(xi, 0.885, "27,075,536", fs=12, w="bold", box=pb)
    t(xi, 0.840, "MS spectra from 2,146 runs", fs=6.2, c=SUB, box=pb)
    tot = 21688803 + 2717217 + 2669516
    rows = (("training", 21688803, INK), ("validation", 2717217, "#9aa5b1"), ("test", 2669516, GREEN))
    xx = xi
    for lab, n, c in rows:
        f = n / tot * wi
        ax.add_patch(Rectangle((xx, 0.785), f - 0.002, 0.026, color=c, zorder=2))
        xx += f
    for i, (lab, n, c) in enumerate(rows):
        y = 0.755 - i * 0.032
        ax.add_patch(Rectangle((xi, y - 0.008), 0.010, 0.016, color=c, zorder=2))
        t(xi + 0.018, y, lab, fs=6.0, box=pb)
        t(xi + wi, y, f"{n:,}", fs=6.0, ha="right", box=pb)
    t(xi, 0.645, "Training partition", fs=6.2, w="bold", box=pb)
    trs = (("TMTpro runs", 18283089, BLUE), ("label-free runs", 2753586, ORANGE))
    xx = xi
    for lab, n, c in trs:
        f = n / (18283089 + 2753586) * wi
        ax.add_patch(Rectangle((xx, 0.598), f - 0.002, 0.026, color=c, zorder=2))
        xx += f
    for i, (lab, n, c) in enumerate(trs):
        y = 0.568 - i * 0.032
        ax.add_patch(Rectangle((xi, y - 0.008), 0.010, 0.016, color=c, zorder=2))
        t(xi + 0.018, y, lab, fs=6.0, box=pb)
        t(xi + wi, y, f"{n:,} spectra", fs=6.0, ha="right", box=pb)
    t(xi, 0.478, "tagged peptides 18,211,829" + NL + "13.0% phosphorylated", fs=6.0, c=SUB, box=pb)
    # ── 모델 ──
    X1, W1 = 0.333, 0.322
    panel(X1, TOP, W1, H, "Model and fine-tuning", BLUE)
    xi, wi = X1 + 0.018, W1 - 0.036
    cb = card(xi, 0.840, wi, 0.085)
    for dx, hgt in ((0.010, .35), (0.018, .75), (0.024, .5), (0.031, 1.0), (0.039, .3),
                    (0.046, .6), (0.054, .45), (0.061, .8), (0.069, .25), (0.076, .4)):
        ax.plot([xi + 0.010 + dx] * 2, [0.852, 0.852 + 0.058 * hgt], color=INK, lw=0.8, zorder=3)
    t(xi + 0.105, 0.896, "MS2 spectrum", fs=6.6, w="bold", box=cb)
    t(xi + 0.105, 0.864, "precursor m/z and charge", fs=6.0, c=SUB, box=cb)
    cb = card(xi, 0.690, wi, 0.095, BLUE)
    t(xi + 0.020, 0.752, "Casanovo 5.2.0", fs=7.0, w="bold", box=cb)
    t(xi + 0.020, 0.718, "encoder-decoder transformer", fs=6.0, c=SUB, box=cb)
    ar((xi + wi / 2, 0.838), (xi + wi / 2, 0.787))
    hw = (wi - 0.012) / 2
    cb = card(xi, 0.455, hw, 0.180, GREEN)
    t(xi + 0.018, 0.608, "Vocabulary", fs=6.6, w="bold", box=cb)
    t(xi + 0.018, 0.540, "[TMTpro]-" + NL + "K[TMTpro]" + NL + "pS pT pY", fs=6.0, fam=MONO, box=cb)
    x2 = xi + hw + 0.012
    cb = card(x2, 0.455, hw, 0.180, GREEN)
    t(x2 + 0.018, 0.608, "Fine-tuning", fs=6.6, w="bold", box=cb)
    t(x2 + 0.018, 0.540, "training partition" + NL + "3,300,000 steps" + NL + "batch 64",
      fs=6.0, c=SUB, box=cb)
    ar((xi + hw / 2, 0.637), (xi + hw / 2, 0.688), GREEN)
    ar((x2 + hw / 2, 0.637), (x2 + hw / 2, 0.688), GREEN)
    # ── 디코딩 ──
    X2, W2 = 0.680, 0.312
    panel(X2, TOP, W2, H, "Decoding", RED)
    xi, wi = X2 + 0.018, W2 - 0.036
    steps = (("Beam search", "ten candidates", None),
             ("Precursor-mass re-ranking", "matches within 20 ppm first", None),
             ("Peptidoform", "[TMTpro]-LApSPEK[TMTpro]", MONO))
    for i, (h1, h2, fam) in enumerate(steps):
        y = 0.825 - i * 0.150
        col = RED if i < 2 else INK
        cb = card(xi, y, wi, 0.100, col)
        ax.add_patch(matplotlib.patches.Ellipse((xi + 0.028, y + 0.050), 0.030, 0.030 * 7.1 / 4.1, color=col, zorder=3))
        t(xi + 0.028, y + 0.050, str(i + 1), fs=6.2, c="white", w="bold", ha="center")
        t(xi + 0.052, y + 0.066, h1, fs=6.4, w="bold", box=cb)
        t(xi + 0.052, y + 0.032, h2, fs=6.0, c=SUB, fam=fam, box=cb)
        if i:
            ar((xi + wi / 2, y + 0.148), (xi + wi / 2, y + 0.102))
    # 칸 사이
    ar((X0 + W0 + 0.003, 0.72), (X1 - 0.003, 0.72), lw=1.2)
    ar((X1 + W1 + 0.003, 0.72), (X2 - 0.003, 0.72), lw=1.2)
    ar((X2 + W2 / 2, TOP - 0.003), (X2 + W2 / 2, 0.402), lw=1.2)
    # ── 평가 ──
    panel(0.008, 0.015, 0.984, 0.385, "Evaluation", GREEN)
    ev = (("Test set", "3,000 TMTpro, 1,000 label-free", "84.7%",
           "TMTpro peptidoform precision" + NL + "label-free 61.2%"),
          ("Four public datasets", "other laboratories, 4,000 spectra each", "69.2-87.4%",
           "peptidoform precision"),
          ("Studies 1-3", "phosphoproteomes not used in training", "+19.7-44.8%",
           "phosphopeptides vs SEQUEST HT"))
    CW, CG = 0.310, 0.017
    for i, (n1, n2, big, cap) in enumerate(ev):
        x = 0.025 + i * (CW + CG)
        cb = card(x, 0.085, CW, 0.250, GREEN)
        t(x + 0.020, 0.305, n1, fs=6.8, w="bold", box=cb)
        t(x + 0.020, 0.272, n2, fs=6.0, c=SUB, box=cb)
        t(x + 0.020, 0.200, big, fs=13, w="bold", c=GREEN, box=cb)
        t(x + 0.020, 0.128, cap, fs=6.0, c=SUB, box=cb)
    t(0.025, 0.048, "Controls:  vocabulary extended but untrained, 0.3% on the test set   ·   "
      "re-ranking disabled, 1.5 to 3.6 points lower on the public datasets", fs=6.0, c=SUB,
      box=(0.008, 0.015, 0.984, 0.385))

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for tt, (x, y, w, h) in checks:
        bb = tt.get_window_extent(r)
        x0, y0 = ax.transData.transform((x, y))
        x1, y1 = ax.transData.transform((x + w, y + h))
        if bb.x0 < x0 - 1 or bb.x1 > x1 - 2 or bb.y0 < y0 or bb.y1 > y1:
            bad += 1
            print("  ! overflow", repr(tt.get_text()[:40]))
    print("modern: overflow", bad)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "FINAL_modern." + ext))
    plt.close(fig)
    plt.rcParams["font.family"] = "DejaVu Sans"


def plain():
    plt.rcParams["font.family"] = "Arial"
    fig, ax = canvas(7.1, 4.1)
    INK, SUB, LINE, BG = "#1b2733", "#5b6670", "#d5dae0", "#f3f5f7"
    BLUE, ORANGE, GREEN, RED = "#2f6f9f", "#d98d2b", "#4a7a3a", "#b8403a"
    MONO = "Consolas"
    checks = []

    def panel(x, y, w, h, title, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.014",
                                    facecolor="white", edgecolor=c, linewidth=0.8,
                                    linestyle=(0, (4, 2.4)), zorder=0))
        ax.text(x + 0.014, y + h - 0.029, title, ha="left", va="center", fontsize=7.4,
                weight="bold", color=c, zorder=2)

    def card(x, y, w, h, c=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.010",
                                    facecolor="white", edgecolor=c or LINE, linewidth=0.7, zorder=1))
        return (x, y, w, h)

    def t(x, y, s, fs=6.2, c=INK, w="normal", ha="left", va="center", fam=None, box=None):
        kw = {"family": fam} if fam else {}
        tt = ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va, zorder=3,
                     linespacing=1.3, **kw)
        if box:
            checks.append((tt, box))
        return tt

    def ar(p1, p2, c="#8a949e", lw=0.9):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=7, linewidth=lw,
                                     color=c, zorder=4))

    TOP, H = 0.43, 0.555
    # ── 코퍼스 ──
    X0, W0 = 0.008, 0.300
    panel(X0, TOP, W0, H, "In-house corpus", C_GREY)
    xi, wi = X0 + 0.022, W0 - 0.044
    pb = (X0, TOP, W0, H)
    t(xi, 0.885, "27,075,536", fs=11, w="bold", box=pb)
    t(xi, 0.845, "MS spectra from 2,146 runs", fs=6.2, c=SUB, box=pb)
    tot = 21688803 + 2717217 + 2669516
    rows = (("training", 21688803, INK), ("validation", 2717217, "#9aa5b1"), ("test", 2669516, GREEN))
    xx = xi
    for lab, n, c in rows:
        f = n / tot * wi
        ax.add_patch(Rectangle((xx, 0.800), f - 0.002, 0.026, color=c, zorder=2))
        xx += f
    for i, (lab, n, c) in enumerate(rows):
        y = 0.772 - i * 0.028
        ax.add_patch(Rectangle((xi, y - 0.008), 0.010, 0.016, color=c, zorder=2))
        t(xi + 0.018, y, lab, fs=6.0, box=pb)
        t(xi + wi, y, f"{n:,}", fs=6.0, ha="right", box=pb)
    t(xi, 0.678, "Training spectra used (21,036,675)", fs=6.2, w="bold", box=pb)
    trs = (("TMTpro runs", 18283089, BLUE), ("label-free runs", 2753586, ORANGE))
    xx = xi
    for lab, n, c in trs:
        f = n / (18283089 + 2753586) * wi
        ax.add_patch(Rectangle((xx, 0.632), f - 0.002, 0.026, color=c, zorder=2))
        xx += f
    for i, (lab, n, c) in enumerate(trs):
        y = 0.604 - i * 0.028
        ax.add_patch(Rectangle((xi, y - 0.008), 0.010, 0.016, color=c, zorder=2))
        t(xi + 0.018, y, lab, fs=6.0, box=pb)
        t(xi + wi, y, f"{n:,} spectra", fs=6.0, ha="right", box=pb)
    # 2026-11-12: TMTpro 표지 펩타이드 스펙트럼의 인산화/비인산화 비율(사용자 지시). 학습 분할 기준.
    t(xi, 0.548, "TMTpro-labeled peptide spectra", fs=6.2, w="bold", box=pb)
    phs = (("phospho", 2375828, "#173a57"), ("non-phospho", 15836001, "#9cc0dc"))
    xx = xi
    for lab, n, c in phs:
        f = n / (2375828 + 15836001) * wi
        ax.add_patch(Rectangle((xx, 0.500), f - 0.002, 0.026, color=c, zorder=2))
        xx += f
    for i, (lab, n, c) in enumerate(phs):
        y = 0.472 - i * 0.028
        ax.add_patch(Rectangle((xi, y - 0.008), 0.010, 0.016, color=c, zorder=2))
        t(xi + 0.018, y, lab, fs=6.0, box=pb)
        t(xi + wi, y, f"{n:,} ({100 * n / (2375828 + 15836001):.1f}%)", fs=6.0, ha="right", box=pb)
    # ── 모델 ──
    X1, W1 = 0.333, 0.322
    panel(X1, TOP, W1, H, "Model and fine-tuning", BLUE)
    xi, wi = X1 + 0.018, W1 - 0.036
    cb = card(xi, 0.840, wi, 0.085)
    for dx, hgt in ((0.010, .35), (0.018, .75), (0.024, .5), (0.031, 1.0), (0.039, .3),
                    (0.046, .6), (0.054, .45), (0.061, .8), (0.069, .25), (0.076, .4)):
        ax.plot([xi + 0.010 + dx] * 2, [0.852, 0.852 + 0.058 * hgt], color=INK, lw=0.8, zorder=3)
    t(xi + 0.105, 0.896, "MS2 spectrum", fs=6.6, w="bold", box=cb)
    t(xi + 0.105, 0.864, "precursor m/z and charge", fs=6.0, c=SUB, box=cb)
    cb = card(xi, 0.690, wi, 0.095)
    t(xi + 0.020, 0.752, "Casanovo 5.2.0", fs=7.0, w="bold", box=cb)
    t(xi + 0.020, 0.718, "encoder-decoder transformer", fs=6.0, c=SUB, box=cb)
    ar((xi + wi / 2, 0.838), (xi + wi / 2, 0.787))
    hw = (wi - 0.012) / 2
    cb = card(xi, 0.455, hw, 0.180)
    t(xi + 0.018, 0.608, "Vocabulary", fs=6.6, w="bold", box=cb)
    t(xi + 0.018, 0.540, "[TMTpro]-" + NL + "K[TMTpro]" + NL + "pS pT pY", fs=6.0, fam=MONO, box=cb)
    x2 = xi + hw + 0.012
    cb = card(x2, 0.455, hw, 0.180)
    t(x2 + 0.018, 0.608, "Fine-tuning", fs=6.6, w="bold", box=cb)
    t(x2 + 0.018, 0.540, "training partition" + NL + "3,300,000 steps" + NL + "batch 64",
      fs=6.0, c=SUB, box=cb)
    ar((xi + hw / 2, 0.637), (xi + hw / 2, 0.688), GREEN)
    ar((x2 + hw / 2, 0.637), (x2 + hw / 2, 0.688), GREEN)
    # ── 디코딩 ──
    X2, W2 = 0.680, 0.312
    panel(X2, TOP, W2, H, "Decoding", RED)
    xi, wi = X2 + 0.018, W2 - 0.036
    steps = (("Beam search", "ten candidates", None),
             ("Precursor-mass re-ranking", "matches within 20 ppm first", None),
             ("Peptidoform", "[TMTpro]-LApSPEK[TMTpro]", MONO))
    for i, (h1, h2, fam) in enumerate(steps):
        y = 0.825 - i * 0.150
        col = RED if i < 2 else INK
        cb = card(xi, y, wi, 0.100)
        t(xi + 0.026, y + 0.066, f"{i + 1}.", fs=6.4, w="bold", ha="center")
        t(xi + 0.044, y + 0.066, h1, fs=6.4, w="bold", box=cb)
        t(xi + 0.044, y + 0.032, h2, fs=6.0, c=SUB, fam=fam, box=cb)
        if i:
            ar((xi + wi / 2, y + 0.148), (xi + wi / 2, y + 0.102))
    # 칸 사이
    ar((X0 + W0 + 0.003, 0.72), (X1 - 0.003, 0.72), lw=1.2)
    ar((X1 + W1 + 0.003, 0.72), (X2 - 0.003, 0.72), lw=1.2)
    ar((X2 + W2 / 2, TOP - 0.003), (X2 + W2 / 2, 0.402), lw=1.2)
    # ── 평가 ──
    panel(0.008, 0.015, 0.984, 0.385, "Evaluation", GREEN)
    ev = (("Test set", "3,000 TMTpro, 1,000 label-free", "84.7%",
           "TMTpro peptidoform precision" + NL + "label-free 61.2%"),
          ("Four public datasets", "other laboratories, 4,000 spectra each", "69.2-87.4%",
           "peptidoform precision"),
          ("Studies 1-3", "phosphoproteomes not used in training", "+19.7-44.8%",
           "phosphopeptides vs SEQUEST HT"))
    CW, CG = 0.304, 0.019
    for i, (n1, n2, big, cap) in enumerate(ev):
        x = 0.025 + i * (CW + CG)
        cb = card(x, 0.085, CW, 0.250)
        t(x + 0.020, 0.305, n1, fs=6.8, w="bold", box=cb)
        t(x + 0.020, 0.272, n2, fs=6.0, c=SUB, box=cb)
        t(x + 0.020, 0.200, big, fs=11, w="bold", c=INK, box=cb)
        t(x + 0.020, 0.128, cap, fs=6.0, c=SUB, box=cb)
    t(0.025, 0.048, "Controls:  vocabulary extended but untrained, 0.3% on the test set   ·   "
      "re-ranking disabled, 1.5 to 3.6 points lower on the public datasets", fs=6.0, c=SUB,
      box=(0.008, 0.015, 0.984, 0.385))

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for tt, (x, y, w, h) in checks:
        bb = tt.get_window_extent(r)
        x0, y0 = ax.transData.transform((x, y))
        x1, y1 = ax.transData.transform((x + w, y + h))
        if bb.x0 < x0 - 1 or bb.x1 > x1 - 2 or bb.y0 < y0 or bb.y1 > y1:
            bad += 1
            print("  ! overflow", repr(tt.get_text()[:40]))
    print("plain: overflow", bad)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "FINAL_plain." + ext))
        fig.savefig(os.path.join(BASE, "figures", "fig0_scheme." + ext))
        fig.savefig(os.path.join(BASE, "figures", "manuscript_v3", "Figure1." + ext))
    plt.close(fig)
    plt.rcParams["font.family"] = "DejaVu Sans"


# ── 최종본 v3 (2026-11-11): 논문 형식 ─────────────────────────────────────
# 흰 바탕, 패널 문자 a-d, 얇은 선, 2-3색. 개념은 그림으로(스펙트럼, 토큰 칸, 후보 재순위),
# 결과는 큰 숫자 대신 작은 표로. 인쇄 폭 7.1 in, 글자 5.8 pt 이상.
def paper():
    plt.rcParams["font.family"] = "Arial"
    fig, ax = canvas(7.1, 3.9)
    INK, SUB, LINE = "#1b2733", "#59636e", "#9aa3ad"
    BLUE, LBLUE, ORANGE, GREEN = "#2f6f9f", "#e3edf5", "#d98d2b", "#4a7a3a"
    MONO = "Consolas"
    checks = []
    XR = 7.1 / 3.9          # x 1 단위가 y 몇 단위인지 (원 등 종횡비 보정)

    def t(x, y, s, fs=6.0, c=INK, w="normal", ha="left", va="center", fam=None, st="normal",
          bound=None):
        kw = {"family": fam} if fam else {}
        tt = ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va, style=st,
                     zorder=3, linespacing=1.3, **kw)
        if bound:
            checks.append((tt, bound))
        return tt

    def head(x, y, letter, title):
        t(x, y, letter, fs=9, w="bold")
        t(x + 0.022, y - 0.002, title, fs=7.0, w="bold")

    def rect(x, y, w, h, fc="white", ec=LINE, lw=0.6, r=0.006, z=1):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

    def ar(p1, p2, c=SUB, lw=0.7, ms=6):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                                     color=c, zorder=4, shrinkA=0, shrinkB=0))

    def legend_rows(x, y, w, rows, bound, dy=0.034):
        for i, (lab, val, c) in enumerate(rows):
            yy = y - i * dy
            ax.add_patch(Rectangle((x, yy - 0.010), 0.009, 0.020, color=c, zorder=2))
            t(x + 0.016, yy, lab, bound=bound)
            t(x + w, yy, val, ha="right", bound=bound)

    def bar(x, y, w, h, parts):
        tot = sum(n for _, n, _ in parts)
        for _, n, c in parts:
            f = n / tot * w
            ax.add_patch(Rectangle((x, y), f - 0.0015, h, color=c, zorder=2))
            x += f

    # ── a. 코퍼스 ──
    A = (0.0, 0.43, 0.30, 0.57)
    head(0.005, 0.975, "a", "In-house corpus")
    t(0.027, 0.912, "27,075,536 MS2 spectra", fs=6.6, w="bold", bound=A)
    t(0.027, 0.878, "from 2,146 runs", fs=6.0, c=SUB, bound=A)
    split = (("training", 21036675, "#33475b"), ("validation", 2717217, "#b5c0cb"),
             ("test", 2669516, GREEN))
    bar(0.027, 0.818, 0.265, 0.034, split)
    legend_rows(0.027, 0.778, 0.265, [(l, f"{n:,}", c) for l, n, c in split], A)
    t(0.027, 0.665, "Training partition", fs=6.2, w="bold", bound=A)
    comp = (("TMTpro runs", 18283089, BLUE), ("label-free runs", 2753586, ORANGE))
    bar(0.027, 0.610, 0.265, 0.034, comp)
    legend_rows(0.027, 0.570, 0.265, [(l, f"{n:,} spectra", c) for l, n, c in comp], A)
    t(0.027, 0.470, "Tagged peptides 18,211,829" + NL + "13.0% of them phosphorylated", fs=5.9, c=SUB, bound=A)

    # ── b. 모델 ──
    B = (0.335, 0.43, 0.32, 0.57)
    head(0.335, 0.975, "b", "Fine-tuned Casanovo")
    # 스펙트럼
    sx, sy = 0.345, 0.775
    ax.plot([sx, sx + 0.085], [sy, sy], color=INK, lw=0.6, zorder=3)
    for dx, hh in ((0.006, .45), (0.014, .9), (0.021, .3), (0.029, .65), (0.036, 1.0),
                   (0.044, .35), (0.052, .55), (0.060, .25), (0.068, .75), (0.076, .4)):
        ax.plot([sx + dx] * 2, [sy, sy + 0.115 * hh], color=INK, lw=0.7, zorder=3)
    t(sx + 0.0425, sy - 0.030, "MS2 spectrum," + NL + "precursor m/z, z", fs=5.8, c=SUB, ha="center",
      va="top", bound=B)
    # 인코더/디코더
    rect(0.460, 0.770, 0.080, 0.125, fc=LBLUE, ec=BLUE)
    t(0.500, 0.8325, "Encoder", fs=6.2, ha="center", bound=(0.460, 0.770, 0.080, 0.125))
    rect(0.565, 0.770, 0.080, 0.125, fc=LBLUE, ec=BLUE)
    t(0.605, 0.8325, "Decoder", fs=6.2, ha="center", bound=(0.565, 0.770, 0.080, 0.125))
    ar((0.435, 0.8325), (0.457, 0.8325))
    ar((0.542, 0.8325), (0.563, 0.8325))
    t(0.5525, 0.912, "transformer, Casanovo 5.2.0", fs=5.8, c=SUB, ha="center", bound=B)
    # 출력 토큰 칸
    toks = (("[TMTpro]-", True, 0.076), ("L", False, 0.020), ("A", False, 0.020), ("pS", True, 0.024),
            ("P", False, 0.020), ("E", False, 0.020), ("K[TMTpro]", True, 0.076))
    tx, ty, th = 0.352, 0.635, 0.050
    ar((0.605, 0.768), (0.605, ty + th + 0.004))
    for s, new, w in toks:
        rect(tx, ty, w, th, fc=LBLUE if new else "white", ec=BLUE if new else LINE, r=0.003)
        t(tx + w / 2, ty + th / 2, s, fs=5.8, fam=MONO, ha="center", bound=(tx, ty, w, th))
        tx += w + 0.004
    t(0.352, 0.585, "Shaded: tokens added to the vocabulary" + NL + "([TMTpro]-, K[TMTpro], pS, pT, pY)",
      fs=5.8, c=SUB, bound=B)
    t(0.352, 0.495, "Fine-tuned end to end on the training partition" + NL
      + "3,300,000 steps, batch size 64", fs=5.9, c=INK, bound=B)

    # ── c. 디코딩 ──
    C = (0.685, 0.43, 0.315, 0.57)
    head(0.685, 0.975, "c", "Precursor-mass re-ranking")
    t(0.700, 0.905, "Beam (10 candidates)", fs=6.0, w="bold", bound=C)
    t(0.860, 0.905, "Re-ranked", fs=6.0, w="bold", bound=C)
    before = (("#1", "412 ppm", False), ("#2", "2 ppm", True), ("#3", "95 ppm", False))
    after = (("#2", "2 ppm", True), ("#1", "412 ppm", False), ("#3", "95 ppm", False))
    for col, rows in ((0.700, before), (0.860, after)):
        for i, (name, d, ok) in enumerate(rows):
            y = 0.835 - i * 0.062
            rect(col, y - 0.024, 0.135, 0.048, fc="#eaf2e6" if ok else "white",
                 ec=GREEN if ok else LINE, r=0.003)
            t(col + 0.008, y, name, fs=5.9, bound=(col, y - 0.024, 0.135, 0.048))
            t(col + 0.127, y, "Δm " + d, fs=5.9, c=GREEN if ok else SUB, ha="right",
              bound=(col, y - 0.024, 0.135, 0.048))
        t(col + 0.0675, 0.835 - 3 * 0.062 + 0.020, "...", fs=7, c=SUB, ha="center")
    ar((0.838, 0.773), (0.857, 0.773))
    t(0.700, 0.570, "Candidates within 20 ppm of the" + NL + "precursor mass are ranked first;" + NL
      + "none is discarded, so every" + NL + "spectrum receives a peptidoform.",
      fs=5.8, c=SUB, va="top", bound=C)

    # ── d. 평가 (표) ──
    D = (0.0, 0.0, 1.0, 0.40)
    head(0.005, 0.385, "d", "Evaluation")
    cols = (0.027, 0.300, 0.530, 0.800)
    hdr = ("Data", "Spectra", "Measure", "Result")
    rows = (("Test set, from the test partition", "3,000 TMTpro; 1,000 label-free",
             "peptidoform precision", "84.7%; 61.2%"),
            ("Four public datasets, other laboratories", "4,000 per dataset",
             "peptidoform precision", "87.4, 83.3, 79.5, 69.2%"),
            ("Studies 1-3, not used in training", "all spectra of each study",
             "phosphopeptides vs SEQUEST HT", "+44.8, +25.1, +19.7%"))
    ax.plot([0.027, 0.990], [0.335, 0.335], color=INK, lw=0.8)
    for x, h in zip(cols, hdr):
        t(x, 0.310, h, fs=6.2, w="bold", bound=D)
    ax.plot([0.027, 0.990], [0.288, 0.288], color=INK, lw=0.5)
    for i, r in enumerate(rows):
        y = 0.255 - i * 0.055
        for j, (x, s) in enumerate(zip(cols, r)):
            t(x, y, s, fs=6.0, w="bold" if j == 3 else "normal", bound=D)
    ax.plot([0.027, 0.990], [0.120, 0.120], color=INK, lw=0.8)
    t(0.027, 0.085, "Controls: vocabulary extended but untrained, 0.3% on the test set; "
      "re-ranking disabled, 1.5 to 3.6 points lower on the public datasets.", fs=5.8, c=SUB, bound=D)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for tt, (x, y, w, h) in checks:
        bb = tt.get_window_extent(r)
        x0, y0 = ax.transData.transform((x, y))
        x1, y1 = ax.transData.transform((x + w, y + h))
        if bb.x0 < x0 - 1 or bb.x1 > x1 + 1 or bb.y0 < y0 - 1 or bb.y1 > y1 + 1:
            bad += 1
            print("  ! overflow", repr(tt.get_text()[:45]))
    print("paper: overflow", bad)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "FINAL_paper." + ext))
    plt.close(fig)
    plt.rcParams["font.family"] = "DejaVu Sans"


if __name__ == "__main__":
    # 인자 없이 = Figure 1 정본(paper). --all = 시안 5종, 인쇄 폭 판, 발표형(modern)도 함께.
    # 인자 없이 = Figure 1 정본(plain). 나머지는 fig1_variants 에만 저장.
    for f in ((v1, v2, v3, v4, v5, final, modern, paper, plain) if "--all" in sys.argv else (plain,)):
        f()
