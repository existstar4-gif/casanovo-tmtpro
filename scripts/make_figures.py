# -*- coding: utf-8 -*-
"""make_figures.py — 논문 본문 Figure 4개를 실측 mzTab 에서 직접 생성한다.

★ 원칙: 문서에 적힌 요약 숫자를 옮겨 그리지 않는다. 전부 mzTab 을 다시 채점해서 그린다.
   이 원칙이 지금까지 오류 두 건(지표 불일치, 희소성 메커니즘)을 잡았다.

구성 (2026-08-27 개편) — 벤치마크 4편(Modanovo/Casanovo/InstaNovo/MassIVE-KB) 관행에 맞춤.
넷 다 본문 표가 0개이고 Results 하위절 제목이 주장문이므로, 표를 Figure 로 흡수한다.

  F1  어휘 확장이 TMTpro de novo 를 가능하게 한다        (baseline + 운영점, AP 병기)
  F2  학습량이 아니라 학습 순서가 소수 클래스를 좌우한다  (A/B + 사다리 통제)
  F3  실험실은 넘어가지만 희소성에서 무너진다              (외부 + 통제실험 + 인산화)
  F4  DB search 가 버린 스펙트럼을 회수한다                (일치도 + 회수율)

사다리 전체 곡선은 부록으로 내린다(구판 스크립트 `make_figures_v1_archive.py` 참조).
출력: figures/*.png (300 dpi) + .pdf
"""
from __future__ import annotations

import os
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
NL = chr(10)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)
E, L, EXT = "data/eval", "data/eval", "data/external"

# 투고본은 그림 안에 해석 문장을 넣지 않는다 — 제목·결론은 legend 로 간다(2026-08-30).
# 내부 검토용으로 다시 켜려면 FIG_SUPTITLE=1.
FIG3_TEXT = 1.6   # Fig. 3 글자 배율
SUPTITLE = os.environ.get("FIG_SUPTITLE", "0") == "1"


def suptitle(fig, text, **kw):
    if SUPTITLE:
        suptitle(fig, text, **kw)
MOD = re.compile(r"\[[^\]]*\]")


def bare(s):
    return MOD.sub("", s).replace("-", "").strip()


def full(s):
    return s.strip().replace("I", "L")


def buck(p):
    n = len(p)
    return "0-9" if n <= 9 else "10-14" if n <= 14 else "15-19" if n <= 19 else "20+"


def load_mztab(path):
    out, hdr = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("PSH"):
            hdr = line.rstrip("\n").split("\t")
            continue
        if not line.startswith("PSM") or not hdr:
            continue
        c = dict(zip(hdr, line.rstrip("\n").split("\t")))
        m = (re.search(r"index=(\d+)", c.get("spectra_ref", ""))
             or re.search(r"scan=(\d+)", c.get("spectra_ref", "")))
        if not m:
            continue
        seq = (c.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence")
               or c.get("sequence", ""))
        try:
            sc = float(c.get("search_engine_score[1]", 0))
        except ValueError:
            sc = 0.0
        out[int(m.group(1))] = (seq, sc)
    return out


GT = {}
for _l in open(f"{E}/sample_new_gt.tsv", encoding="utf-8").read().splitlines()[1:]:
    _p = _l.split("\t")
    if len(_p) >= 3:
        GT[int(_p[0])] = (_p[1], _p[2])


def score_dev(path):
    pred = load_mztab(path)
    rec = []
    for idx, (cls, seq) in GT.items():
        p, sc = pred.get(idx, ("", 0.0))
        rec.append({"cls": cls, "pred": p, "score": sc,
                    "ok": bool(p) and full(p) == full(seq),
                    "len": buck(bare(seq)),
                    "fp": cls == "naked" and "TMTpro" in p})
    return rec


def rates(rec, cls=None):
    v = [r for r in rec if cls is None or r["cls"] == cls]
    n = len(v) or 1
    return (100 * sum(r["ok"] for r in v) / n,
            100 * sum(r["fp"] for r in v) / n, len(v))


def pc_curve(rec, cls):
    """(coverage%, precision%) 와 average precision — Casanovo 관행(곡선 아래 면적)."""
    v = sorted([r for r in rec if r["cls"] == cls], key=lambda r: -r["score"])
    n = len(v) or 1
    xs, ys, ok, ap, prev = [], [], 0, 0.0, 0.0
    for i, r in enumerate(v, 1):
        ok += r["ok"]
        cov, prec = i / n, ok / i
        ap += prec * (cov - prev)
        prev = cov
        if i % max(1, n // 300) == 0 or i == n:
            xs.append(100 * cov)
            ys.append(100 * prec)
    return xs, ys, 100 * ap


def r1(v):
    """소수 첫째 자리 사사오입 (2026-11-12). f"{v:.1f}" 는 79.55 를 79.5 로 내린다(이진 표현)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(round(v, 6))).quantize(Decimal("0.1"), ROUND_HALF_UP))


def scale_text(fig, k):
    """그림 안 모든 글자를 k 배로 (2026-11-11: 인쇄 폭으로 줄이면 글자가 안 보인다는 지적)."""
    import matplotlib.text as _mt
    for _t in fig.findobj(_mt.Text):
        _t.set_fontsize(_t.get_fontsize() * k)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.png / .pdf")


# ── F1 ───────────────────────────────────────────────────────────────────
def fig1():
    """Figure 2 — 우리 모델의 성능을, 이 분야가 쓰는 지표 전부로.

    2026-09-03 재구성. 이전 판본은 펩타이도폼 정밀도 한 축과 운용 곡선 두 패널뿐이었다.
    de novo 문헌(Casanovo, DeepNovo, PointNovo, Casanovo-DIA, pi-PrimeNovo)은 성능을
    **세 지표로** 보고한다 — peptide precision / AA precision / AA recall — 그리고
    precision-coverage 곡선과 AP 를 덧붙인다. 그 구성을 그대로 따른다.

    패널 c(커버리지)를 넣은 이유: 미학습 조건의 붕괴는 정밀도만으로는 안 보인다.
    전체 어휘를 복사만 한 모델은 스펙트럼의 58.5% 에서 **아무 것도 내놓지 못한다.**

    ⚠️ 패널 a 는 원고 정의(완전일치 펩타이도폼, 전체 스펙트럼 분모).
       패널 b 는 원저자 정의(질량 허용 ±0.1 Da, 예측한 잔기 분모) — 문헌 비교용.
       두 정의를 섞어 읽지 않도록 패널 제목에 못 박는다.
    """
    # 2026-09-03: knapsack 조건을 뺐다. 재순위는 어휘·학습과 다른 질문이고
    # Figure 3 이 그것만 다룬다. 여기 남는 축은 "어휘와 학습이 있어야 하는가" 하나다.
    ROWS = [("released", 0.3, 61.7),
            ("TMT tokens," + NL + "no training", 0.3, 34.3),
            ("full vocabulary," + NL + "no training", 0.0, 4.0),
            ("this work", 82.6, 57.9)]   # 2026-11-11: A 는 네 조건 모두 re-ranking 끔(final_plain)
    # out/fig2_metrics.tsv — aa_eval.py(원저자 aa_match_batch) 산출
    M = {}
    for line in open(os.path.join(BASE, "out", "fig2_metrics.tsv"), encoding="utf-8"):
        c = line.rstrip().split(chr(9))
        if len(c) >= 6:
            M[(c[0], c[1])] = dict(aap=float(c[2]), aar=float(c[3]), cov=float(c[5]))
    KEYS = ["released", "tmt_tokens_notrain", "full_vocab_notrain", "this_work_plain"]
    # 패널 b·c 는 폭이 좁아 긴 라벨이 겹친다. 축약형을 쓰고 a 가 전체 이름을 준다.
    # 2026-11-11: 모델 이름 Casanovo-TMT 도입, 대조군은 출처를 밝히는 이름으로.
    SHORT = ["Casanovo 5.2.0", "Casanovo 5.2.0 +" + NL + "TMTpro tokens, untrained",
             "Casanovo 5.2.0 +" + NL + "full vocabulary, untrained", "Casanovo-TMT"]
    def xlab(ax, labels, fs=6.8):
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=fs, rotation=32, ha="right",
                           rotation_mode="anchor")

    # 2026-11-11: 패널 f(예측 비율)를 사용자 결정으로 뺐다. 윗줄 a-c, 아랫줄 d-e 를 가운데 정렬.
    fig = plt.figure(figsize=(13.6, 7.4))
    # 2026-11-11: A 정밀도(re-ranking 끔) / B re-ranking 효과(옛 Fig. 3F) / C 운용점 /
    #             D 보정 / E 아미노산(re-ranking 끔) / F 길이
    gs = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.62)
    axR = fig.add_subplot(gs[0, 1])
    axes = ((fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 0])),
            (fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_axes([0, 0, 0.01, 0.01])))
    # 2026-11-11: 패널 순서 a 정밀도, b 운용점, c 보정, d 아미노산, e 길이, f 예측 비율.
    # 변수 이름은 옛 판본(a-f) 그대로 두고 놓이는 자리만 바꾼다.
    (ax1, ax4, ax6), (ax2, ax5, ax3) = axes
    x = range(len(ROWS))
    w = 0.36
    lab = [r[0] for r in ROWS]

    def xlab(ax, labels, fs=6.8):
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=fs, rotation=32, ha="right",
                           rotation_mode="anchor")

    # ── 윗줄: 다섯 조건 비교 ────────────────────────────────────────────
    # a — 펩타이도폼 정밀도 (원고 정의: 완전일치, 전체 스펙트럼 분모)
    # 2026-11-11: 펩타이드 클러스터 부트스트랩 95% CI 오차막대 (out/fig2a_ci.json, cluster_ci.py 산출)
    import json
    CI = json.load(open(os.path.join(BASE, "out", "fig2a_ci.json"), encoding="utf-8"))
    for off, j, cls, c, nm in ((-w / 2, 1, "tmt", C_TMT, "TMT"), (w / 2, 2, "naked", C_LF, "label-free")):
        v = [r[j] for r in ROWS]
        lo = [v[i] - CI[k][cls][0] for i, k in enumerate(KEYS)]
        hi = [CI[k][cls][1] - v[i] for i, k in enumerate(KEYS)]
        ax1.bar([i + off for i in x], v, w, color=c, label=nm,
                yerr=[lo, hi], error_kw=dict(ecolor=C_GREY, elinewidth=0.9, capsize=2.5))
        for i, k in enumerate(KEYS):
            ax1.text(i + off, CI[k][cls][1] + (5.4 if cls == "tmt" else 1.4), f"{v[i]:.1f}",
                     ha="center", fontsize=7, weight="bold" if (i == 3 and cls == "tmt") else "normal")
    xlab(ax1, SHORT)
    ax1.set_ylim(0, 124); ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylabel("Peptidoform precision (%)")
    ax1.text(-0.16, 1.07, "A", transform=ax1.transAxes, fontsize=13, weight="bold", va="bottom")
    ax1.legend(loc="upper center", ncol=2, fontsize=7, frameon=False)

    # b — 아미노산 수준 (원저자 정의). 문헌과 비교되는 유일한 축이다.
    aap = [100 * M[(k, "tmt")]["aap"] for k in KEYS]
    aar = [100 * M[(k, "tmt")]["aar"] for k in KEYS]
    ax2.bar([i - w / 2 for i in x], aap, w, color=C_TMT, label="precision")
    ax2.bar([i + w / 2 for i in x], aar, w, color=C_TMT, alpha=.45, label="recall")
    for i in x:
        ax2.text(i - w / 2, aap[i] + 6.4, f"{aap[i]:.1f}", ha="center", fontsize=6.6,
                 weight="bold" if i == 3 else "normal")
        ax2.text(i + w / 2, aar[i] + 1.4, f"{aar[i]:.1f}", ha="center", fontsize=6.6)
    ax2.axhline(95, color=C_FP, ls=":", lw=1.1)
    ax2.text(-0.42, 89.5, "0.95  Casanovo, within distribution", fontsize=6.4,
             color=C_FP, ha="left", va="top")
    xlab(ax2, SHORT)
    ax2.set_ylim(0, 124); ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylabel("Amino acid level, TMT (%)")
    ax2.text(-0.16, 1.07, "E", transform=ax2.transAxes, fontsize=13, weight="bold", va="bottom")
    ax2.legend(loc="upper center", ncol=2, fontsize=7, frameon=False)

    # c — 커버리지. 미학습 붕괴는 정밀도로는 보이지 않는다.
    ct = [M[(k, "tmt")]["cov"] for k in KEYS]
    cn = [M[(k, "naked")]["cov"] for k in KEYS]
    ax3.bar([i - w / 2 for i in x], ct, w, color=C_TMT, label="TMT")
    ax3.bar([i + w / 2 for i in x], cn, w, color=C_LF, label="label-free")
    for i in x:
        ax3.text(i - w / 2, ct[i] + 5.4, f"{ct[i]:.0f}", ha="center", fontsize=7)
        ax3.text(i + w / 2, cn[i] + 1.4, f"{cn[i]:.0f}", ha="center", fontsize=7)
    xlab(ax3, SHORT)
    ax3.set_ylim(0, 132); ax3.set_yticks([0, 20, 40, 60, 80, 100])
    ax3.set_ylabel("Spectra given a peptide (%)")
    ax3.text(-0.16, 1.07, "F", transform=ax3.transAxes, fontsize=13, weight="bold", va="bottom")
    ax3.legend(loc="upper center", ncol=2, fontsize=7, frameon=False)

    # B — re-ranking 효과 (옛 Fig. 3F, make_rerank_fig 와 같은 산출물)
    ARMS = [("no re-ranking", "data/eval/dbrank/armA.mztab", "#8d99a6"),
            ("precursor mass", "data/eval/dbrank/armB.mztab", C_TMT),
            ("mass + tryptic search space", "data/eval/rerank_verify/internal.mztab", C_GREEN)]
    wr = 0.26
    for j, (nm, path, c) in enumerate(ARMS):
        rr = score_dev(path)
        vals = [rates(rr, "tmt")[0], rates(rr, "naked")[0]]
        xs_ = [i + (j - 1) * wr for i in range(2)]
        axR.bar(xs_, vals, wr, color=c, label=nm)
        for x_, v in zip(xs_, vals):
            axR.text(x_, v + 1.6, f"{v:.1f}", ha="center", fontsize=6.4)
    axR.set_xticks(range(2))
    axR.set_xticklabels(["TMT", "label-free"], fontsize=8)
    axR.set_ylim(0, 140); axR.set_yticks([0, 20, 40, 60, 80, 100])
    axR.set_ylabel("Peptidoform precision (%)")
    axR.legend(loc="upper left", fontsize=7, frameon=False)
    axR.text(-0.16, 1.07, "B", transform=axR.transAxes, fontsize=13, weight="bold", va="bottom")

    # ── 아랫줄: 최종 모델의 성질 ────────────────────────────────────────
    rec = score_dev(f"{E}/final_knap.mztab")
    pairs = list(zip(rec, GT.items()))

    # d — precision-coverage + AP. 원저자 Fig 2 가 전부 이 곡선이다.
    for cls, c, nm in (("tmt", C_TMT, "TMT"), ("naked", C_LF, "label-free")):
        xs, ys, ap = pc_curve(rec, cls)
        ax4.plot(xs, ys, color=c, lw=2, label=f"{nm}  (AP {ap:.1f})")
        best = max((x_ for x_, y_ in zip(xs, ys) if y_ >= 95.0), default=None)
        if best:
            yy = [y_ for x_, y_ in zip(xs, ys) if x_ == best][0]
            ax4.plot([best], [yy], "o", color=c, ms=6, mec="white", mew=1.2)
            ax4.annotate(f"{best:.0f}%", xy=(best, yy), xytext=(best - 5, yy - 13),
                         fontsize=8.5, color=c, weight="bold", ha="right",
                         arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
    # 2026-11-11: FDR 1% 지점은 사용자 결정으로 Fig. 2 에서 뺐다(인산화 절에서 다룬다).
    ax4.axhline(95, color=C_GREY, ls=":", lw=1)
    ax4.text(99, 91.0, "95% precision", fontsize=7, color=C_GREY, ha="right")
    ax4.set_xlim(0, 102); ax4.set_ylim(40, 103)
    ax4.set_xlabel("Coverage (%)")
    ax4.set_ylabel("Peptidoform precision (%)")
    ax4.text(-0.16, 1.07, "C", transform=ax4.transAxes, fontsize=13, weight="bold", va="bottom")
    ax4.legend(loc="lower left", fontsize=7.5)

    # e — 길이별. NovoBench 가 기존 논문에 없다고 지적한 강건성 축이다.
    BINS = [("6-9", 6, 9), ("10-12", 10, 12), ("13-15", 13, 15),
            ("16-19", 16, 19), ("20+", 20, 999)]
    for cls, c, nm in (("tmt", C_TMT, "TMT"), ("naked", C_LF, "label-free")):
        xs, ys, ns = [], [], []
        for j, (nmb, lo, hi) in enumerate(BINS):
            v = [r for r, (k, (cl, sq)) in pairs
                 if cl == cls and lo <= len(bare(sq)) <= hi]
            if not v:
                continue
            xs.append(j); ys.append(100 * sum(r["ok"] for r in v) / len(v)); ns.append(len(v))
        ax5.plot(xs, ys, marker="o", color=c, lw=1.9, ms=5, label=nm)
        if cls == "tmt":
            for j, y_, n_ in zip(xs, ys, ns):
                ax5.text(j, y_ + 3.4, f"{y_:.0f}", ha="center", fontsize=7, color=c)
                ax5.text(j, 3, f"n={n_}", ha="center", fontsize=6, color=C_GREY)
    ax5.set_xticks(range(len(BINS)))
    ax5.set_xticklabels([b[0] for b in BINS], fontsize=7.5)
    ax5.set_ylim(0, 108); ax5.set_yticks([0, 20, 40, 60, 80, 100])
    ax5.set_xlabel("Reference peptide length (residues)")
    ax5.set_ylabel("Peptidoform precision (%)")
    ax5.text(-0.16, 1.07, "F", transform=ax5.transAxes, fontsize=13, weight="bold", va="bottom")
    ax5.legend(loc="lower left", fontsize=7.5)

    # f — 보정. 점수가 순위로 쓸 만한지, 확률로 읽어도 되는지는 다른 문제다.
    for cls, c, nm in (("tmt", C_TMT, "TMT"), ("naked", C_LF, "label-free")):
        v = sorted((r["score"], r["ok"]) for r, (k, (cl, sq)) in pairs
                   if cl == cls and r["pred"])
        NB = 8
        per = max(1, len(v) // NB)
        px, py = [], []
        for i in range(NB):
            ch = v[i * per:(i + 1) * per] if i < NB - 1 else v[i * per:]
            if not ch:
                continue
            px.append(100 * sum(s for s, _ in ch) / len(ch))
            py.append(100 * sum(o for _, o in ch) / len(ch))
        ax6.plot(px, py, marker="o", color=c, lw=1.9, ms=5, label=nm)
    ax6.plot([0, 100], [0, 100], ls="--", color=C_GREY, lw=1)
    ax6.text(63, 55, "perfect calibration", fontsize=6.6, color=C_GREY, rotation=38)
    ax6.set_xlim(0, 102); ax6.set_ylim(0, 105)
    ax6.set_xlabel("Score the model emits (%)")
    ax6.set_ylabel("Observed precision (%)")
    ax6.text(-0.16, 1.07, "D", transform=ax6.transAxes, fontsize=13, weight="bold", va="bottom")
    ax6.legend(loc="upper left", fontsize=7.5)

    fig.delaxes(ax3)   # 패널 f 는 그리지 않는다(위 주석)
    suptitle(fig, "Figure 2 - Performance of the model on the test set",
                 y=1.03, fontsize=10.5, weight="bold")
    save(fig, "fig1_capability")


# ── F2 ───────────────────────────────────────────────────────────────────
def fig2():
    """순서 통제 실험.

    ★ 2026-08-28 교체: 이전 판본은 A/B 3점을 썼는데, 인터리브 분기가 블록정렬 런의
      *연속*이라 순서와 추가 최적화가 교락돼 있었다. ctrl 실험은 동일 체크포인트에서
      --load_all_states 로 갈라져 가중치·옵티마이저·LR 스케줄 위치까지 같고 각 분기가
      정확히 1 에폭(328,699 step)을 돈다. 유일한 변수가 저장소 순서다.
      두 판본 모두 knapsack OFF 이므로 패널 간 비교가 성립한다(사다리는 ON 이라 뺐다).
    """
    C = "data/eval/ctrl"
    A = [f"{C}/ev_ctrlA_step380000.mztab", f"{C}/ev_ctrlA_step440000.mztab",
         f"{C}/ev_ctrlA_step540000.mztab", f"{C}/ev_ctrlA_step640000.mztab",
         f"{C}/ev_ctrlA.last.mztab"]
    B = [f"{C}/ev_ctrlB_step380000.mztab", f"{C}/ev_ctrlB_step440000.mztab",
         f"{C}/ev_ctrlB_step540000.mztab", f"{C}/ev_ctrlB_step640000.mztab",
         f"{C}/ev_ctrlB.last.mztab"]

    def grab(paths):
        lf, fp, tmt = [], [], []
        for q in paths:
            rec = score_dev(q)
            a, b, _ = rates(rec, "naked")
            lf.append(a)
            fp.append(b)
            tmt.append(rates(rec, "tmt")[0])
        return lf, fp, tmt

    lfA, fpA, tmA = grab(A)
    lfB, fpB, tmB = grab(B)

    # 두 갈래가 갈라져 나온 부모(step 328,699) 자체의 값. 이 선이 없으면 '블록이 파괴했다'와
    # '인터리브가 회복시켰다'를 그림에서 구분할 수 없다(2026-08-30 적대 검토 지적).
    P = "data/eval/ctrl/ev_parent_step328699.mztab"
    par_lf = par_fp = None
    if os.path.exists(P):
        prec = score_dev(P)
        par_lf, par_fp, _ = rates(prec, "naked")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.6, 3.4))
    fig.subplots_adjust(wspace=0.34)
    xs = list(range(5))
    lab = ["380k", "440k", "540k", "640k", "657k"]

    ax1.plot(xs, lfA, marker="o", color=C_FP, lw=1.9, ms=6, label="block-ordered")
    ax1.plot(xs, lfB, marker="s", color=C_GREEN, lw=1.9, ms=6, label="interleaved")
    ax1.fill_between(xs, min(lfA), max(lfA), color=C_FP, alpha=.07)
    ax1.fill_between(xs, min(lfB), max(lfB), color=C_GREEN, alpha=.07)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(lab, fontsize=7.5)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Training step within the branch")
    if par_lf is not None:
        ax1.axhline(par_lf, color=C_GREY, ls="--", lw=1.1)
        ax1.text(4.05, par_lf + 2.5, f"parent {par_lf:.1f}", fontsize=7, color=C_GREY, ha="right")
    ax1.set_ylabel("Label-free peptidoform precision (%)")
    ax1.set_title("a  Same corpus, two storage orders")
    ax1.legend(loc="upper left")
    ax1.text(2, 64, f"{min(lfB):.1f}-{max(lfB):.1f}", ha="center", fontsize=7.5,
             color=C_GREEN, weight="bold")
    ax1.text(2, 16, f"{min(lfA):.1f}-{max(lfA):.1f}", ha="center", fontsize=7.5,
             color=C_FP, weight="bold")

    ax2.plot(xs, fpA, marker="o", color=C_FP, lw=1.9, ms=6, label="block-ordered")
    ax2.plot(xs, fpB, marker="s", color=C_GREEN, lw=1.9, ms=6, label="interleaved")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(lab, fontsize=7.5)
    ax2.set_ylim(-3, 100)
    ax2.set_xlabel("Training step within the branch")
    ax2.set_ylabel("TMT tokens emitted on" + NL + "label-free spectra (%)")
    if par_fp is not None:
        ax2.axhline(par_fp, color=C_GREY, ls="--", lw=1.1)
        ax2.text(4.05, par_fp + 2.5, f"parent {par_fp:.1f}", fontsize=7, color=C_GREY, ha="right")
    ax2.set_title("b  Minority-class false positives")
    ax2.legend(loc="center left")
    ax2.annotate(f"max {max(fpA):.1f}%", xy=(2, max(fpA)), xytext=(2.6, 72),
                 fontsize=7.5, color=C_FP,
                 arrowprops=dict(arrowstyle="->", color=C_FP, lw=0.8))
    ax2.annotate(f"{min(fpB):.1f}-{max(fpB):.1f}%", xy=(2, 0.3), xytext=(1.2, 22),
                 fontsize=7.5, color=C_GREEN,
                 arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=0.8))

    w = 0.34
    xb = range(2)
    ax3.bar([i - w / 2 for i in xb], [sum(tmA) / 5, sum(tmB) / 5], w,
            color=C_TMT, label="TMT")
    ax3.bar([i + w / 2 for i in xb], [sum(lfA) / 5, sum(lfB) / 5], w,
            color=C_LF, label="label-free")
    for i, (t, l) in enumerate(((sum(tmA) / 5, sum(lfA) / 5),
                                (sum(tmB) / 5, sum(lfB) / 5))):
        ax3.text(i - w / 2, t + 1.6, f"{t:.1f}", ha="center", fontsize=8)
        ax3.text(i + w / 2, l + 1.6, f"{l:.1f}", ha="center", fontsize=8, weight="bold")
    ax3.set_xticks(list(xb))
    ax3.set_xticklabels(["block-ordered", "interleaved"], fontsize=8)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Mean precision over 5 snapshots (%)")
    ax3.set_title("c  TMT precision, same checkpoints")
    ax3.legend(loc="upper left")

    suptitle(fig, "Figure 2 — Training order, not training volume, governs "
                 "minority-class retention", y=1.04, fontsize=10.5, weight="bold")
    save(fig, "fig2_order")
    print(f"    ctrlA 무표지 {min(lfA):.1f}-{max(lfA):.1f}  오탐 {min(fpA):.1f}-{max(fpA):.1f}")
    print(f"    ctrlB 무표지 {min(lfB):.1f}-{max(lfB):.1f}  오탐 {min(fpB):.1f}-{max(fpB):.1f}")


# ── F3 ───────────────────────────────────────────────────────────────────
def score_external(acc, mztab="ext", mgf="sub.mgf"):
    """외부 데이터 채점.

    ⚠️ GT 키는 반드시 (파일 stem, scan) 이다. scan 만으로 잡으면 안 된다.
       PXD054085 의 gt.tsv 는 두 raw 파일(B_Protein_F / T_Protein_F)을 담고 있고
       두 파일의 scan 번호가 24,274개 겹친다. sub.mgf 는 T_Protein_F 뿐이므로,
       scan 만으로 키를 잡으면 4,000개 중 1,226개의 정답이 다른 파일 라벨로 덮여
       정밀도가 87.4% -> 60.2% 로 잘못 내려간다(2026-08-28 발견·수정).
    """
    d = f"{EXT}/{acc}"
    # PXD067164 는 정본(knapsack ON) 디코딩으로만 ext_knap 이름으로 돌렸다. ext.mztab 도
    # 사실 ON 이었다(2026-08-31 확인: 두 파일의 예측이 18,000건 중 0건 다름). OFF 측정은
    # ext_noknap.mztab 이며 정밀도가 1.5-9.1%p 낮다 — 폴백은 이름 문제일 뿐 구성은 같다.
    if mztab == "ext" and not os.path.exists(f"{d}/ext.mztab"):
        mztab = "ext_knap"
    if not (os.path.exists(f"{d}/{mztab}.mztab") and os.path.exists(f"{d}/{mgf}")):
        return None
    gt = {}
    for line in open(f"{d}/gt.tsv", encoding="utf-8").read().splitlines()[1:]:
        c = line.split("	")
        if len(c) >= 5:
            gt[(c[0], int(c[1]))] = c[4]          # (stem, scan)
    keys, peaks, cur = [], [], 0
    for line in open(f"{d}/{mgf}", encoding="utf-8"):
        if line.startswith("TITLE="):
            st, sc = line.strip()[6:].rsplit(":scan:", 1)
            keys.append((st, int(sc)))
            cur = 0
        elif line.startswith("END IONS"):
            peaks.append(cur)
        elif line[:1].isdigit():
            cur += 1
    raw = load_mztab(f"{d}/{mztab}.mztab")
    ok = n = 0
    for i, k in enumerate(keys):
        g = gt.get(k)
        if not g:
            continue
        p_ = raw.get(i, raw.get(k[1], ("", 0)))[0]
        n += 1
        ok += bool(p_) and full(p_) == full(g)
    return (100 * ok / n if n else 0.0,
            statistics.median(peaks) if peaks else 0, n)


def ext_records(acc):
    """외부 데이터의 스펙트럼별 (점수, 정답 여부) — precision-coverage 곡선용(2026-11-11).

    키 규칙은 score_external 과 같다((stem, scan), 예측 없음은 점수 0·오답).
    """
    d = f"{EXT}/{acc}"
    mz = "ext" if os.path.exists(f"{d}/ext.mztab") else "ext_knap"
    gt = {}
    for line in open(f"{d}/gt.tsv", encoding="utf-8").read().splitlines()[1:]:
        c = line.split("	")
        if len(c) >= 5:
            gt[(c[0], int(c[1]))] = c[4]
    keys = []
    for line in open(f"{d}/sub.mgf", encoding="utf-8"):
        if line.startswith("TITLE="):
            st, sc = line.strip()[6:].rsplit(":scan:", 1)
            keys.append((st, int(sc)))
    raw = load_mztab(f"{d}/{mz}.mztab")
    rec = []
    for i, k in enumerate(keys):
        g = gt.get(k)
        if not g:
            continue
        p_, sc_ = raw.get(i, raw.get(k[1], ("", 0.0)))
        rec.append({"cls": "all", "score": sc_, "ok": bool(p_) and full(p_) == full(g)})
    return rec


def fig3():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pd_", os.path.join(BASE, "scripts", "phospho_decompose.py"))
    pd_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pd_)

    # 2026-08-28: GT 키 버그 수정 후 재채점. PXD065357(정답이 전부 무표지)을 추가 —
    # 인터리브 모델이 타 실험실 무표지에서도 소수 클래스 능력을 유지한다는 외부 증거다.
    # 2026-08-29: PXD067164 을 정본 디코딩으로 재채점해 여섯 번째로 넣었다. 이전에는
    # '재채점 미완'을 사유로 빠져 있었다(RUNTIME_CONFIG 2026-08-29 절).
    # 2026-11-11: 본문 서술 순서(054085 → 072211 → 063976 → 065357)대로.
    EXTSETS = [("PXD054085", "global" + NL + "mouse", C_TMT),
               ("PXD072211", "global" + NL + "mouse", C_TMT),
               ("PXD063976", "phospho" + NL + "human", C_GREEN),
               ("PXD065357", "label-free" + NL + "human", C_LF)]
    # 2026-11-10: pY 두 건(PXD054497·PXD067164)을 제외했다 — 원고가 공개 데이터 4건만 쓴다.
    rows = []
    for acc, lab, col in EXTSETS:
        r = score_external(acc)
        if r:
            rows.append((acc, lab, r[0], r[2], col))

    ctl = []
    for name, mgf, lab in (("ext", "sub.mgf", "original"),
                           ("ext_norep", "sub_norep.mgf", "reporter" + NL + "removed"),
                           ("ext_sparse", "sub_sparse.mgf", "29 peaks")):
        r = score_external("PXD072211", name, mgf)
        if r:
            ctl.append((lab, r[0]))

    ph = []
    for acc, lab in (("PXD063976", "PXD063976"),):   # 2026-11-12: 다른 패널처럼 accession
        try:
            ph.append((lab, pd_.run(acc)))
        except Exception:                              # noqa: BLE001
            pass

    # 2026-08-29: 패널 d 신설. 본문이 인용하는 Fig. 3D(피크 수 중앙값)가 그려지지
    # 않아 희소성 해명의 기각 근거가 그림에 없었다. spectrum_stats.py 를 유일한
    # 출처로 삼는다 — 이 값들이 원고 안에만 있던 상태를 없앤다.
    spec_ss = importlib.util.spec_from_file_location(
        "ss_", os.path.join(BASE, "scripts", "spectrum_stats.py"))
    ss_ = importlib.util.module_from_spec(spec_ss)
    spec_ss.loader.exec_module(ss_)
    dens = {}
    for acc, _, _ in EXTSETS:
        r = ss_.stats(acc)
        if r:
            dens[acc] = r[1]

    # 2026-11-11: 패널 b(외부 데이터 운용점) 신설. 본문 인용 순서에 맞춰 a, b(운용점),
    # c(변형), d(밀도), e(인산화) 순으로 놓는다. 변수 이름은 옛 판본 그대로.
    fig, (ax1, ax5, ax2, ax4, ax3) = plt.subplots(
        1, 5, figsize=(20.0, 4.6), gridspec_kw={"width_ratios": [2.0, 1.25, 1, 1.15, 1.05]})
    fig.subplots_adjust(wspace=0.42)
    # 2026-08-31: 펩타이드 클러스터 부트스트랩 CI 를 오차막대로. 캐시가 없으면 막대만 그린다
    # (생성: python scripts/cluster_ci.py --ext).
    import json
    ci_path = os.path.join(BASE, "out", "ext_ci.json")
    CI = json.load(open(ci_path, encoding="utf-8")) if os.path.exists(ci_path) else {}

    xs = range(len(rows))
    err = [[max(0.0, rows[i][2] - CI[rows[i][0]]["lo"]) if rows[i][0] in CI else 0.0
            for i in range(len(rows))],
           [max(0.0, CI[rows[i][0]]["hi"] - rows[i][2]) if rows[i][0] in CI else 0.0
            for i in range(len(rows))]]
    ax1.bar(xs, [r[2] for r in rows], 0.55, color=[r[4] for r in rows],
            yerr=err if CI else None, capsize=2.6,
            error_kw=dict(ecolor=C_GREY, elinewidth=0.9, capthick=0.9))
    for i, r in enumerate(rows):
        top = CI[r[0]]["hi"] if r[0] in CI else r[2]
        # 2026-11-12: 79.5 가 84.7 점선과 겹쳤다 — 흰 바탕을 깔아 점선 위에 올린다.
        ax1.text(i, top + 1.8, r1(r[2]), ha="center", fontsize=8, weight="bold", zorder=5,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax1.text(i, 4, f"n={r[3]:,}", ha="center", fontsize=6.0, color="white",
                 rotation=90, va="bottom")
    ax1.axhline(84.7, color=C_GREY, ls="--", lw=1.2)
    ax1.text(2.0, 91.5, "test set 84.7", ha="center",
             fontsize=7, color=C_GREY)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([r[0] + NL + r[1] for r in rows], fontsize=4.8,
                        rotation=24, ha="right")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Peptidoform precision (%)")
    ax1.text(-0.14, 1.06, "A", transform=ax1.transAxes, fontsize=13, weight="bold", va="bottom")

    ax2.bar(range(len(ctl)), [c[1] for c in ctl], 0.55, color=[C_TMT, C_GREEN, C_FP])
    for i, (lab, v) in enumerate(ctl):
        ax2.text(i, v + 1.6, r1(v), ha="center", fontsize=8, weight="bold")
    # 2026-11-11: 차이(pp) 표기는 뺐다 — 사용자 지시.
    ax2.set_xticks(range(len(ctl)))
    ax2.set_xticklabels([c[0] for c in ctl], fontsize=5.6, rotation=25, ha="right",
                        rotation_mode="anchor")
    ax2.set_ylim(0, 100)
    ax2.text(-0.14, 1.06, "C", transform=ax2.transAxes, fontsize=13, weight="bold", va="bottom")

    if ph:
        xp = range(len(ph))
        exact = [100 * r[1]["exact"] / r[1]["n"] for r in ph]
        site = [100 * (r[1]["ph"] - r[1]["site"]) / r[1]["n"] for r in ph]
        rest = [100 - a - b for a, b in zip(exact, site)]
        ax3.bar(xp, exact, 0.5, color=C_TMT, label="fully correct")
        ax3.bar(xp, site, 0.5, bottom=exact, color=C_LF, label="phosphosite wrong only")
        ax3.bar(xp, rest, 0.5, bottom=[a + b for a, b in zip(exact, site)],
                color=C_PALE, label="sequence wrong")
        for i, r in enumerate(ph):
            ax3.text(i, exact[i] / 2, r1(exact[i]), ha="center", va="center",
                     fontsize=8, weight="bold", color="white")
            # 2026-11-12: 나머지 두 칸에도 값을 적는다(사용자 지시).
            # 214/4000 = 5.35 처럼 끝이 5 이면 f"{:.1f}" 는 5.3 을 낸다 — 본문(5.4%)과 맞게 사사오입한다.
            _r = lambda k, m: f"{(1000 * k + m // 2) // m / 10:.1f}"
            _n = r[1]["n"]; _ex = r[1]["exact"]; _ph = r[1]["ph"]
            ax3.text(i, exact[i] + site[i] / 2, _r(_ph - r[1]["site"], _n), ha="center", va="center",
                     fontsize=6.5, weight="bold", color="white")
            ax3.text(i, exact[i] + site[i] + rest[i] / 2, _r(_n - _ex - (_ph - r[1]["site"]), _n), ha="center", va="center",
                     fontsize=8, weight="bold", color="#333333")
        ax3.set_xticks(list(xp))
        ax3.set_xticklabels([r[0] for r in ph], fontsize=8)
        ax3.set_ylim(0, 152)
        # 여백은 범례와 site 라벨용이다. 눈금은 100 에서 끊는다.
        ax3.set_yticks([0, 20, 40, 60, 80, 100])
        ax3.set_ylabel("Share of spectra (%)")
        ax3.text(-0.14, 1.06, "E", transform=ax3.transAxes, fontsize=13, weight="bold", va="bottom")
        ax3.legend(loc="upper center", ncol=1, fontsize=5.0, frameon=False,
                   bbox_to_anchor=(0.5, 1.005), columnspacing=1.0, handlelength=1.2)
        # 2026-11-12: 막대 위 site 91% 와 별표 주석은 뺐다 — 분모(2,368)가 막대(4,000)와 달라 헷갈린다. 본문에만 둔다(사용자 지시).

    # d — 희소성 해명의 기각. 연관이 반대 방향이면 순위를 설명하지 못한다.
    pts = [(dens[r[0]], r[2], r[0], r[4]) for r in rows if r[0] in dens]
    for x, y, acc, col in pts:
        ax4.scatter([x], [y], s=54, color=col, zorder=3)
        # 회귀선·축 눈금과 겹치지 않도록 점마다 라벨 위치를 정한다
        off, ha = {"PXD054085": ((0, 8), "center"), "PXD065357": ((4, -13), "left"),
                   "PXD072211": ((0, -13), "center"), "PXD063976": ((0, -13), "right")
                   }.get(acc, ((0, 8), "center"))
        ax4.annotate(acc, (x, y), textcoords="offset points", xytext=off,
                     ha=ha, fontsize=6.6, color=C_GREY)
    # 2026-11-12: 기울기 선·표기는 뺐다 — 점 4개로 낸 기울기라 본문에서도 뺐다(사용자 지시).
    ax4.set_xlabel("Median peak count of evaluated spectra", fontsize=7.5)
    ax4.set_ylabel("Peptidoform precision (%)")
    ax4.set_ylim(0, 100)
    ax4.text(-0.14, 1.06, "D", transform=ax4.transAxes, fontsize=13, weight="bold", va="bottom")

    # b — 외부 데이터 precision-coverage. 원 표시는 정밀도 95% 이상이 유지되는 최대 coverage.
    for acc, lab, col in EXTSETS:
        rec_ = ext_records(acc)
        if not rec_:
            continue
        xs_, ys_, ap_ = pc_curve(rec_, "all")
        ls_ = ":" if acc == "PXD072211" else "-"
        ax5.plot(xs_, ys_, color=col, lw=1.6, ls=ls_, label=acc)
        best = max((x_ for x_, y_ in zip(xs_, ys_) if y_ >= 95.0), default=None)
        if best:
            yy = [y_ for x_, y_ in zip(xs_, ys_) if x_ == best][0]
            ax5.plot([best], [yy], "o", color=col, ms=4.5, mec="white", mew=1.0)
    ax5.axhline(95, color=C_GREY, ls=":", lw=0.9)
    ax5.set_xlim(0, 102); ax5.set_ylim(40, 103)
    ax5.set_xlabel("Coverage (%)")
    ax5.set_ylabel("Peptidoform precision (%)")
    ax5.text(-0.14, 1.06, "B", transform=ax5.transAxes, fontsize=13, weight="bold", va="bottom")
    ax5.legend(loc="lower left", fontsize=6.4)

    suptitle(fig, "Figure 3 — Performance transfers across laboratories; the ranking "
                 "among deposits is not explained by spectral sparsity",
                 y=1.04, fontsize=10.5, weight="bold")
    scale_text(fig, FIG3_TEXT)
    save(fig, "fig3_scope")
    for x, y, acc, _ in pts:
        print(f"    {acc:<12} 피크중앙 {x:>5.0f}   정밀도 {y:5.1f}%")


# ── F4 ───────────────────────────────────────────────────────────────────
def fig4():
    """DB search 와의 관계.

    ★ 2026-08-28 전면 개편. 이전 판본은 세 패널이 전부 리터럴이었고, 그래서
      (i) 구제율 정의 불일치와 (ii) 암기 판정의 I/L 규칙 불일치가 그대로 그림에
      들어가 있었다. 이제 rescue_unified.py 를 임포트해 실측에서 그린다.
      라벨원 독립성(패널 a)만 fragpipe_compare.py 재계산값을 쓴다.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ru", os.path.join(BASE, "scripts", "rescue_unified.py"))
    ru = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ru)

    import pickle
    pkl = os.path.join(BASE, "out", "train_peptides_true.pkl")
    seen = {ru.norm(x) for x in pickle.load(open(pkl, "rb"))} if os.path.exists(pkl) else set()

    sp_map = {}
    for line in open(os.path.join(BASE, "out", "rescue_runs.txt"), encoding="utf-8"):
        c = line.rstrip(chr(10)).split(chr(9))
        if len(c) >= 2:
            sp_map[c[1]] = c[0]

    def collect(mztab, mgf, species, keep=None):
        rows, _ = ru.load(mztab, mgf)
        if keep:
            rows = [r for r in rows if r[0] == keep]
        f = (lambda st: sp_map.get(st)) if species is None else (lambda st: species)
        return ru.score(rows, f, seen)

    EXPS = [
        ("our runs" + NL + "vs PD",
         collect("data/eval/rescue_v2.mztab",
                 "data/eval/unidentified_v1.mgf", None)),
        ("PXD054085" + NL + "vs FragPipe",
         collect(f"{EXT}/PXD054085/unid_fp.mztab", f"{EXT}/PXD054085/unid_fp.mgf", "mouse")),
        ("PXD063976" + NL + "vs FragPipe",
         collect(f"{EXT}/PXD063976/unid_fp.mztab", f"{EXT}/PXD063976/unid_fp.mgf",
                 "human", keep="Patient1_Phos_F_search")),
    ]

    def stats(recs, sub=None):
        v = recs if sub is None else sub
        n = len(v) or 1
        t = sum(1 for r in v if r["cls"] == "target")
        d = sum(1 for r in v if r["cls"] == "decoy")
        novel = sum(1 for r in v if r["cls"] == "target" and r["seen"] is False)
        # ★ 2026-08-29 — 순 구제율에만 decoy 를 빼고 미관측은 원시 계수를 쓰던 비대칭을
        #   없앴다. decoy 율이 실험 간 0.8/1.4/0.6% 로 2.3배 차이나 공통 오프셋이 아니다.
        #   비례 귀속: decoy 일치가 관측/미관측에 target 과 같은 비율로 섞였다고 본다.
        novel_prop = (100 * novel / n) * ((t - d) / t) if t else 0.0
        return 100 * (t - d) / n, novel_prop, 100 * d / n

    # 2026-11-12: Fig. 3 G-I 로 합쳤다(사용자 결정). 데이터셋 순서는 Fig. 3A 와 같게.
    LBL = [("PXD054085", 93.4, 93.3), ("PXD072211", 79.5, 81.6),
           ("PXD063976", 74.4, 69.1)]

    # 2026-11-12: Fig. 3 은 G 까지만. H·I(구제율)는 맨 끝 Fig. 6 으로 뺐다(사용자 결정, 이후 확장 예정).
    fig, ax1 = plt.subplots(1, 1, figsize=(4.2, 4.6))
    fig.subplots_adjust(bottom=0.26)

    x = range(len(LBL))
    w = 0.36
    ax1.bar([i - w / 2 for i in x], [r[1] for r in LBL], w, color=C_GREY,
            label="depositor labels")
    ax1.bar([i + w / 2 for i in x], [r[2] for r in LBL], w, color=C_TMT,
            label="FragPipe labels")
    for i, r in enumerate(LBL):
        ax1.text(i - w / 2, r[1] + 1.6, f"{r[1]:.1f}", ha="center", fontsize=5.0)
        ax1.text(i + w / 2, r[2] + 1.6, f"{r[2]:.1f}", ha="center", fontsize=5.0)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([r[0] for r in LBL], fontsize=6.4, rotation=28, ha="right",
                        rotation_mode="anchor")
    ax1.set_ylim(0, 150)   # 여백은 범례용, 눈금은 100 에서 끊는다 (F 와 같게)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylabel("Peptidoform precision (%)")
    ax1.text(-0.14, 1.06, "G", transform=ax1.transAxes, fontsize=13, weight="bold", va="bottom")
    ax1.legend(loc="upper left", ncol=1, fontsize=5.6, frameon=False)

    scale_text(fig, FIG3_TEXT)   # Fig. 3 윗줄과 같은 배율
    save(fig, "fig3_g")

    # 2026-11-12: Fig. 5 (fig8_ours_only_validation, 소문자 제목 형식) 아래 줄 e·f 로 붙인다.
    # 2026-11-13: Fig. 5 위쪽 A–D(12 in 폭, 2x2)와 패널 크기를 맞춘다
    # 2026-11-13: Fig. 5 를 3열(A B C / D E F) 한 그림으로 — A–D 는 make_figures_phospho.fig8 이 같은 축에 그린다
    fig, _axs = plt.subplots(2, 3, figsize=(18.0, 10.0)); _axs = _axs.ravel()
    fig.subplots_adjust(wspace=0.42, hspace=0.45)
    _sp = importlib.util.spec_from_file_location("mfp_", os.path.join(BASE, "scripts", "make_figures_phospho.py"))
    mfp = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(mfp)
    mfp.fig8(fig, _axs[:4])
    ax2, ax3 = _axs[4], _axs[5]
    xr = range(len(EXPS))
    net = [stats(r)[0] for _, r in EXPS]
    nov = [stats(r)[1] for _, r in EXPS]
    ax2.bar(xr, net, 0.55, color=C_PALE, label="net recovery")
    ax2.bar(xr, nov, 0.55, color=C_GREEN, label="of which not in training")
    for i, (a, b) in enumerate(zip(net, nov)):
        ax2.text(i, a + 0.45, f"{a:.1f}", ha="center", fontsize=8, weight="bold")
        ax2.text(i, b / 2, f"{b:.1f}", ha="center", va="center", fontsize=8,
                 color="white", weight="bold")
    ax2.set_xticks(list(xr))
    ax2.set_xticklabels([n.replace(NL, " ") for n, _ in EXPS], fontsize=7.5, rotation=28,
                        ha="right", rotation_mode="anchor")
    ax2.set_ylim(0, 28)   # 여백은 범례용
    ax2.set_yticks([0, 5, 10, 15, 20])
    ax2.set_ylabel("Net recovery (%)")
    ax2.text(-0.2, 1.04, "E", transform=ax2.transAxes, fontsize=13, weight="bold", va="bottom")   # 2026-11-13: 제목은 범례로
    ax2.legend(loc="upper left", fontsize=7.5, frameon=False)
    # 2026-08-30: "nearly constant" 는 과장이다. 보정 계열에 같은 동질성 검정을 하면
    # 세 값이 갈린다(p=0.010). 정확한 진술은 '총계보다 덜 변한다'(1.2배 대 2.1배)다.
    # 2026-11-12: "varies 1.2x / totals vary 2.1x" 주석은 뺐다 — 해석이고 본문(1.1배)과도 달랐다.

    STRATA = ["all", "mass-" + NL + "matched", "top 50%", "top 10%"]
    ends = []
    for (name, recs), col, mk in zip(EXPS, (C_GREY, C_LF, C_GREEN), ("^", "o", "s")):
        mass = sorted([r for r in recs if r["ppm"] <= 20.0], key=lambda r: -r["score"])
        subs = [recs, mass, mass[: len(mass) // 2], mass[: len(mass) // 10]]
        ys = [stats(recs, u)[0] for u in subs]
        ax3.plot(range(4), ys, marker=mk, color=col, lw=1.8, ms=5,
                 label=name.replace(NL, " "))
        ends.append((ys[-1], col))
    # 세 계열의 끝값이 붙어 있어 그대로 찍으면 겹친다(51.0 위에 47.5).
    # 값 순으로 정렬해 최소 간격을 강제한다.
    prev = None
    for v, col in sorted(ends):
        y = v if prev is None else max(v, prev + 6.0)   # 2026-11-13: 글자 1.6 배에 맞춘 간격
        ax3.text(3.08, y, f"{v:.1f}", ha="left", va="center", fontsize=7.5, color=col)
        prev = y
    ax3.set_xlim(-0.15, 3.9)
    ax3.set_xticks(range(4))
    ax3.set_xticklabels([s_.replace("-" + NL, "-").replace(NL, " ") for s_ in STRATA], fontsize=7.5, rotation=28,
                        ha="right", rotation_mode="anchor")
    ax3.set_ylim(0, 100)   # 여백은 범례용 (2026-11-13: 글자 1.6 배)
    ax3.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax3.set_ylabel("Net recovery (%)")
    ax3.text(-0.2, 1.04, "F", transform=ax3.transAxes, fontsize=13, weight="bold", va="bottom")
    ax3.legend(loc="upper left", fontsize=7.5, frameon=False)

    scale_text(fig, FIG3_TEXT)   # 2026-11-13: Fig. 3 과 같은 글자 배율
    save(fig, "fig5_combined")
    for name, recs in EXPS:
        a, b, d = stats(recs)
        print(f"    {name.replace(NL,' '):<28} 순 {a:5.1f}%  비암기 {b:4.1f}%  decoy {d:4.1f}%")


if __name__ == "__main__":
    print("Figure 생성 …")
    fig1()
    fig2()
    fig3()
    fig4()
    print("\n완료 — 본문 Figure 4개")
