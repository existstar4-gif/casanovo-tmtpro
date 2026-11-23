# -*- coding: utf-8 -*-
"""make_supplementary.py — 보충 자료(Supplementary Figures S1–S3, Tables S2–S4)를 원자료에서 직접 만든다 (2026-11-14).

make_figures.py / make_figures_phospho.py 의 원칙을 따른다: 요약 숫자를 옮겨 적지 않고 로그·JSON·설정 파일을 읽는다.
그림 안에 제목·해석 문장은 넣지 않는다(범례가 설명). 스터디는 Study 1–7 로만 표기한다.

  Fig. S1  최종 학습 런의 학습·검증 손실 (data/runs/mixed_v11.log)
  Fig. S2  공개 TMTpro 데이터 3건에서 원저자 라벨과 FragPipe 라벨의 일치 (fragpipe_compare.py 와 같은 규칙)
  Fig. S3  방법별 유일 인산화 펩타이드와 합의 인산화 펩타이드의 같은 자리 리포터 상관 (same_site_all_methods.json)
  Table S3 학습·추론 구성 (casanovo_tmt_v11.yaml, casanovo_tmt_eval_b10.yaml, 환경변수)
  Table S4 Fig. 4D 의 인산화 펩타이드 수 (harvest_gain.json)
  Table S2 검색 엔진 매개변수 (PD 내보내기 관측값, FragPipe workflow/fragger.params/로그, Sage config.json)

    python scripts/make_supplementary.py            # 전부
    python scripts/make_supplementary.py S1 T4      # 일부
출력: figures/supplementary/*.png (600 dpi) + .pdf, out/supplementary/*.tsv + .xlsx
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
# 본문 그림과 같은 글꼴·색·패널 표기를 쓴다 (import 시 rcParams 도 같이 설정된다)
import make_figures_phospho as mfp   # noqa: E402
from make_figures_phospho import (C_TMT, C_LF, C_GREY, C_GREEN, C_PALE,   # noqa: E402,F401
                                  DISPLAY, STUDIES, METHODS, panel, scale_text)

plt.rcParams.update({"savefig.dpi": 600})
FIG_OUT = os.path.join(BASE, "figures", "supplementary")
TAB_OUT = os.path.join(BASE, "out", "supplementary")
os.makedirs(FIG_OUT, exist_ok=True)
os.makedirs(TAB_OUT, exist_ok=True)
V = os.path.join(BASE, "out", "quant", "validation")
NL = chr(10)

TRAIN_LOG = "data/runs/mixed_v11.log"
STORE_CHANGE = 2_360_000     # 파일명 순서 → 인터리브 저장본 (Experimental Procedures)
FINAL_STEP = 3_300_000       # 보고에 쓴 체크포인트 (mixed_v11.final_step3300000.ckpt)
PD_DIR = "data/pd/"
PD_BASES = {"study1": "study1_phospho_fr_0",
            "study2": "study2_phospho_fr_",
            "study3": "study3_phospho_fr_"}
FP_RUN = "data/fragpipe"
FP_STUDY = {"study1": "s1", "study2": "s2", "study3": "s3"}
PUBLIC = ["PXD054085", "PXD072211", "PXD063976"]   # 본문 Fig. 3G 순서
DEPOSITOR_ENGINE = {"PXD054085": "Mascot", "PXD072211": "Mascot and SEQUEST HT", "PXD063976": "Mascot"}


def save_fig(fig, name):
    import print_fit   # 인쇄 폭 6.5 in 에서 글자 7 pt 이상 (2026-11-18, C26)
    print_fit.save(fig, FIG_OUT, name, 6.5)
    plt.close(fig)
    print(f"  figures/supplementary/{name}.png / .pdf")


def save_table(df, name, sheet):
    df.to_csv(os.path.join(TAB_OUT, f"{name}.tsv"), sep="\t", index=False, encoding="utf-8")
    with pd.ExcelWriter(os.path.join(TAB_OUT, f"{name}.xlsx"), engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name=sheet, index=False)
        ws = xw.sheets[sheet]
        for col in ws.columns:   # 열 너비를 내용에 맞춘다(최대 70자)
            w = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(70, max(10, w + 2))
        from openpyxl.styles import Alignment, Font
        for c in ws[1]:
            c.font = Font(bold=True)
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
    print(f"  out/supplementary/{name}.tsv / .xlsx  ({len(df)} rows)")


# ───────────────────────────── Fig. S1 학습 곡선 ─────────────────────────────
def read_history(path=TRAIN_LOG):
    """model._log_history 행 → {step: (train, val)}. 재개로 같은 step 이 두 번 찍히면 나중 값을 쓴다."""
    hist = {}
    pat = re.compile(r"model\._log_history : (\d+)\s+(\S+)\s+(\S+)")
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if m:
            tr, va = (float(x) if x != "nan" else np.nan for x in m.group(2, 3))
            prev = hist.get(int(m.group(1)), (np.nan, np.nan))
            hist[int(m.group(1))] = (tr if np.isfinite(tr) else prev[0], va if np.isfinite(va) else prev[1])
    return hist


def figS1():
    hist = read_history()
    steps = np.array(sorted(hist))
    tr = np.array([hist[s][0] for s in steps]); va = np.array([hist[s][1] for s in steps])
    keep = steps <= FINAL_STEP
    after = steps[~keep]
    steps, tr, va = steps[keep], tr[keep], va[keep]
    mv, mt = np.isfinite(va), np.isfinite(tr)
    rows = pd.DataFrame({"step": steps, "train_loss": tr, "val_loss": va})
    rows.to_csv(os.path.join(TAB_OUT, "figS1_source_loss.tsv"), sep="\t", index=False)
    print(f"  validation points {mv.sum()}, training points {mt.sum()}, steps {steps[0]:,}–{steps[-1]:,}"
          f" (log rows after {FINAL_STEP:,} dropped: {list(after)})")
    for s, t in zip(steps[mt], tr[mt]):
        print(f"    train  step {s:>9,}  {t:.4f}")
    pre = (steps < STORE_CHANGE) & mv; post = (steps > STORE_CHANGE) & mv
    print(f"    val  before change: median {np.median(va[pre]):.3f}, range {va[pre].min():.3f}–{va[pre].max():.3f}")
    print(f"    val  after  change: median {np.median(va[post]):.3f}, range {va[post].min():.3f}–{va[post].max():.3f};"
          f" last {va[mv][-1]:.4f} at {steps[mv][-1]:,}")

    fig, ax = plt.subplots(figsize=(6.6, 3.2))   # 2026-11-18 (C26): 인쇄 크기
    ax.plot(steps[mv] / 1e6, va[mv], color=C_TMT, lw=1.1, label="validation loss (every 5,000 steps)")
    ax.plot(steps[mt] / 1e6, tr[mt], "o", color=C_LF, ms=6, mec="white", mew=0.8, zorder=4,
            label="training loss (mean over each pass)")
    ax.axvline(STORE_CHANGE / 1e6, color=C_GREY, lw=1.2, ls="--",
               label="step 2,360,000: file-name order → interleaved")
    ax.set_yscale("log")
    ax.set_ylim(0.2, 3.2)
    ax.set_yticks([0.25, 0.5, 1, 2]); ax.set_yticklabels(["0.25", "0.5", "1", "2"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0, FINAL_STEP / 1e6 * 1.01)
    ax.set_xlabel("Training step (millions)")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend(loc="upper right", ncol=1, fontsize=7, framealpha=0.9, frameon=True, edgecolor="none")
    ax.set_ylim(0.2, 5.0)
    fig.tight_layout()
    save_fig(fig, "figS1_training_loss")


# ───────────────────────────── Fig. S2 라벨 일치 ─────────────────────────────
def label_agreement():
    """fragpipe_compare.run() 과 같은 분모: 우리 평가 부분집합 중 FragPipe 도 동정하고, 우리 예측과 원저자 라벨이 있는 스펙트럼."""
    import fragpipe_compare as fc
    out = {}
    for acc in PUBLIC:
        fp = fc.load_fragpipe(acc); ours = fc.load_ours(acc); gt = fc.load_gt(acc)
        both = [k for k in ours if k in fp]
        cnt = collections.Counter(); sub = collections.Counter()
        for k in both:
            if not ours.get(k) or not gt.get(k):
                continue
            g, f = fc.peptidoform(gt[k]), fc.peptidoform(fp[k][0])
            if g == f:
                cnt["same"] += 1
            elif fc.bare(g) == fc.bare(f):
                cnt["site"] += 1
                # 세부: 인산화 자리만 다른가, 그 밖의 수식(N말단 표지·아세틸·산화 등)이 다른가
                strip_p = lambda s: s.replace("[Phospho]", "")
                sub["phosphosite only" if strip_p(g) == strip_p(f) else "other modification"] += 1
            else:
                cnt["seq"] += 1
        n = sum(cnt.values())
        out[acc] = dict(n_both=len(both), n=n, same=cnt["same"], site=cnt["site"], seq=cnt["seq"],
                        site_phospho_only=sub["phosphosite only"], site_other=sub["other modification"])
        print(f"  {acc}: both {len(both):,}, scored {n:,}; same {cnt['same']:,} ({100*cnt['same']/n:.1f}%),"
              f" same seq diff site/mod {cnt['site']:,} ({100*cnt['site']/n:.1f}%; phospho-only {sub['phosphosite only']},"
              f" other {sub['other modification']}), diff seq {cnt['seq']:,} ({100*cnt['seq']/n:.1f}%)")
    return out


def figS2():
    d = label_agreement()
    pd.DataFrame([{"dataset": a, "depositor engine": DEPOSITOR_ENGINE[a], "identified by both engines": v["n_both"],
                   "scored (depositor label readable)": v["n"],
                   "same peptidoform": v["same"], "same peptidoform (%)": round(100 * v["same"] / v["n"], 1),
                   "same sequence, different site or modification": v["site"],
                   "  of which phosphosite only": v["site_phospho_only"],
                   "  of which other modification": v["site_other"],
                   "same sequence, different site or modification (%)": round(100 * v["site"] / v["n"], 1),
                   "different sequence": v["seq"], "different sequence (%)": round(100 * v["seq"] / v["n"], 1)}
                  for a, v in d.items()]).to_csv(os.path.join(TAB_OUT, "figS2_source_label_agreement.tsv"),
                                                 sep="\t", index=False)
    CATS = [("same", "same peptidoform", C_TMT), ("site", "same sequence, different site or modification", "#8fb3d1"),
            ("seq", "different sequence", C_PALE)]
    fig, ax = plt.subplots(figsize=(6.0, 4.6))   # 2026-11-18 (C26): 인쇄 크기
    for i, acc in enumerate(PUBLIC):
        v = d[acc]; bottom = 0
        hl = acc == "PXD063976"
        side_y = []   # 작은 조각의 값은 막대 오른쪽에 적는다(겹치지 않게 위로 밀어낸다)
        for key, lab, col in CATS:
            pct = 100 * v[key] / v["n"]
            ax.bar(i, pct, 0.62, bottom=bottom, color=col, edgecolor="black" if hl else "white",
                   linewidth=1.4 if hl else 0.5, label=lab if i == 0 else None, zorder=3)
            if pct >= 10:
                ax.text(i, bottom + pct / 2, f"{pct:.1f}%", ha="center", va="center", fontsize=7,
                        color="white" if col == C_TMT else "black", zorder=4)
            else:
                y = bottom + pct / 2
                if side_y and y - side_y[-1] < 4.5:
                    y = side_y[-1] + 4.5
                side_y.append(y)
                ax.annotate(f"{pct:.1f}%", (i + 0.31, bottom + pct / 2), xytext=(i + 0.36, y - 6),
                            fontsize=6.5, va="center", ha="left",
                            arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.6))
            bottom += pct
        ax.text(i, 101, f"n = {v['n']:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(PUBLIC)))
    ax.set_xticklabels(PUBLIC)
    for t in ax.get_xticklabels():
        if t.get_text() == "PXD063976":
            t.set_fontweight("bold")
    ax.set_xlim(-0.5, len(PUBLIC) - 0.25)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Spectra (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.10), ncol=1, fontsize=7)
    fig.tight_layout()
    fig.tight_layout()
    save_fig(fig, "figS2_label_agreement")
    return d


# ───────────────────────────── Fig. S3 같은 자리 상관 ─────────────────────────────
def figS3():
    d = json.load(open(os.path.join(V, "same_site_all_methods.json"), encoding="utf-8"))
    groups = [(m, f"{m} unique", c) for m, c in METHODS] + \
             [("consensus (>=2 methods)", "consensus (≥2 methods)", "white")]
    rows = []
    for tag, _, name in STUDIES:
        for key, lab, _ in groups:
            v = d[tag][key]
            rows.append({"study": name, "group": lab,
                         "phosphopeptides in group": v.get("unique", v.get("n")),
                         "with a same-site partner (n)": v["with_partner"],
                         "with a same-site partner (%)": round(100 * v["with_partner"] / v.get("unique", v.get("n")), 1),
                         "mean Pearson r": v["mean_r"]})
    src = pd.DataFrame(rows)
    src.to_csv(os.path.join(TAB_OUT, "figS3_source_same_site.tsv"), sep="\t", index=False)
    print(src.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.6, 3.6))   # 2026-11-18 (C26): 인쇄 크기
    w = 0.16; x = np.arange(len(STUDIES))
    for j, (key, lab, col) in enumerate(groups):
        r = [d[t][key]["mean_r"] for t, _, _ in STUDIES]
        n = [d[t][key]["with_partner"] for t, _, _ in STUDIES]
        xs = x + (j - 2) * w
        ax.bar(xs, r, w * 0.92, color=col, label=lab, edgecolor=C_GREY if col == "white" else "white",
               linewidth=0.9 if col == "white" else 0.5, hatch="///" if col == "white" else None, zorder=3)
        for xi, ri, ni in zip(xs, r, n):
            ax.text(xi, ri + 0.012, f"{ni:,}", ha="center", va="bottom", fontsize=5.6, rotation=90)
    ax.axhline(0, color=C_GREY, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([n for _, _, n in STUDIES])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean same-site reporter correlation (r)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    fig.tight_layout()
    fig.tight_layout()
    save_fig(fig, "figS3_same_site_all_methods")


# ───────────────────────────── Table S2 구성 ─────────────────────────────
DESC = {
    "min_peptide_len": "Shortest peptide predicted or used in training (residues)",
    "max_peptide_len": "Longest peptide predicted or used in training (residues)",
    "predict_batch_size": "Spectra per batch at inference; also the validation batch size during training",
    "top_match": "Predictions reported per spectrum",
    "accelerator": "Compute device",
    "devices": "Device selection (empty = automatic)",
    "n_beams": "Beam width: candidate peptides retained per spectrum during decoding",
    "precursor_mass_tol": "Precursor mass tolerance (ppm) used by Casanovo to flag predictions that do not match the precursor",
    "isotope_error_range": "Isotope errors considered with the precursor mass tolerance",
    "enzyme": "Database-search mode only; not used for de novo sequencing",
    "digestion": "Database-search mode only; not used for de novo sequencing",
    "missed_cleavages": "Database-search mode only; not used for de novo sequencing",
    "max_mods": "Database-search mode only; not used for de novo sequencing",
    "allowed_fixed_mods": "Database-search mode only; not used for de novo sequencing",
    "allowed_var_mods": "Database-search mode only; not used for de novo sequencing",
    "random_seed": "Random seed",
    "n_log": "Logging frequency",
    "tb_summarywriter": "TensorBoard logging",
    "log_metrics": "Per-step metric logging",
    "log_every_n_steps": "Logging interval (steps)",
    "lance_dir": "Directory of the indexed spectrum store built from the training files",
    "val_check_interval": "Training steps between validation-loss evaluations",
    "min_peaks": "Minimum number of peaks for a spectrum to be used",
    "max_peaks": "Maximum number of most intense peaks retained per spectrum",
    "min_mz": "Lowest m/z retained",
    "max_mz": "Highest m/z retained",
    "min_intensity": "Peaks below this fraction of the base peak are removed",
    "remove_precursor_tol": "Peaks within this distance of the precursor m/z are removed (m/z)",
    "max_charge": "Highest precursor charge; fixed by the charge embedding of the released checkpoint",
    "dim_model": "Transformer model dimension (must match the released checkpoint)",
    "n_head": "Attention heads",
    "dim_feedforward": "Feed-forward dimension",
    "n_layers": "Transformer layers in encoder and decoder",
    "dropout": "Dropout rate",
    "dim_intensity": "Intensity encoding dimension (empty = same as model dimension)",
    "warmup_iters": "Linear learning-rate warm-up (steps)",
    "cosine_schedule_period_iters": "Period of the cosine learning-rate schedule (steps)",
    "learning_rate": "Peak learning rate",
    "weight_decay": "Weight decay (L2 penalty applied by Adam)",
    "train_label_smoothing": "Label smoothing of the token cross-entropy",
    "train_batch_size": "Spectra per training batch",
    "max_epochs": "Maximum passes over the training partition",
    "shuffle": "Shuffle training spectra",
    "shuffle_buffer_size": "Spectra held in the shuffle buffer",
    "num_sanity_val_steps": "Validation batches run before training",
    "calculate_precision": "Compute precision during validation",
    "accumulate_grad_batches": "Batches per optimizer step",
    "gradient_clip_val": "Gradient clipping threshold",
    "gradient_clip_algorithm": "Gradient clipping method",
    "precision": "Numerical precision of training and inference",
    "replace_isoleucine_with_leucine": "Isoleucine written as leucine in references and predictions",
    "massivekb_tokenizer": "Use the MassIVE-KB peptide notation",
}
# 학습에만 의미가 있는 키: 추론 설정의 값은 쓰이지 않으므로 'not used' 로 적는다
TRAIN_ONLY = {"random_seed", "n_log", "tb_summarywriter", "log_metrics", "log_every_n_steps", "lance_dir",
              "val_check_interval", "warmup_iters", "cosine_schedule_period_iters", "learning_rate", "weight_decay",
              "train_label_smoothing", "train_batch_size", "max_epochs", "shuffle", "shuffle_buffer_size",
              "num_sanity_val_steps", "calculate_precision", "accumulate_grad_batches", "gradient_clip_val",
              "gradient_clip_algorithm"}
NEW_TOKENS = {"S[Phospho]", "T[Phospho]", "Y[Phospho]", "K[TMTpro]", "[TMTpro]-", "R[Phospho]", "K[Phospho]"}


def fmt(v):
    if v is None:
        return "(empty)"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, list):
        return ", ".join(fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}" if (v == 0 or abs(v) >= 1e-3) else f"{v:.1e}"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    return str(v)


def tableS2():
    tr = yaml.safe_load(open(os.path.join(BASE, "casanovo_tmt_v11.yaml"), encoding="utf-8"))
    inf = yaml.safe_load(open(os.path.join(BASE, "casanovo_tmt_eval_b10.yaml"), encoding="utf-8"))
    rows = []

    def add(section, key, t, i, desc):
        rows.append({"Section": section, "Key": key, "Training value": t, "Inference value": i, "Description": desc})

    for k in tr:
        if k in ("residues", "new_token_init"):
            continue
        tv = "(local path)" if k == "lance_dir" else fmt(tr[k])
        if k in TRAIN_ONLY:
            iv = "not used"
        else:
            iv = fmt(inf.get(k))
        sec = "Configuration file"
        add(sec, k, tv, iv, DESC.get(k, ""))
    missing = set(inf) - set(tr)
    assert not missing, missing
    for tok, m in tr["residues"].items():
        assert abs(inf["residues"][tok] - m) < 1e-9, tok
        new = tok in NEW_TOKENS
        init = tr["new_token_init"].get(tok)
        desc = ("Added token; output-layer row initialized from " + init) if new else "Token of the released checkpoint"
        add("Residue vocabulary (residues)", tok, f"{m:.6f}", f"{m:.6f}", desc + " (mass in Da)")
    # 환경변수 (수정한 casanovo/denovo/model.py 가 읽는다; config 와 체크포인트에는 저장되지 않는다)
    env = [
        ("CASANOVO_TMT_MARGIN", "5", "not set (0)", "Margin of the auxiliary hinge term between the logits of [TMTpro]- and the stop token at positions whose target is [TMTpro]-; 0 disables the term"),
        ("CASANOVO_TMT_LAMBDA", "1.0", "not used", "Weight of the hinge term in the training loss; the validation loss excludes it"),
        ("CASANOVO_KNAPSACK", "not set (0)", "1", "Precursor-mass re-ranking: beam candidates whose peptidoform mass matches the measured precursor are ranked first; no candidate is discarded"),
        ("CASANOVO_KNAPSACK_PPM", "not used", "20", "Tolerance of precursor-mass re-ranking (ppm), one carbon-13 isotope allowed"),
        ("CASANOVO_DBRANK", "not set (0)", "1 for Studies 1–3 (Figs. 4 and 5) and for the tryptic search-space condition of Figs. 2B and 3F; not set elsewhere", "Tryptic search space re-ranking: within each mass group, candidates present in the tryptic digest of the reference proteome are ranked first"),
        ("CASANOVO_DBRANK_INDEX", "not used", "Tryptic digest index of the reference proteome of the species, mouse or human (up to two missed cleavages, 6–50 residues, I = L, including peptides starting after an initial Met)", "Digest used by CASANOVO_DBRANK"),
        ("CASANOVO_DBRANK_BONUS", "not used", "1.00 (default)", "Score added to candidates present in the tryptic search space"),
        ("CASANOVO_GATE", "not set (0)", "not set (0)", "Reporter-ion gating of TMTpro tokens; not used in this work"),
    ]
    for k, t, i, dsc in env:
        add("Environment variable", k, t, i, dsc)
    # 실행 사실
    run = [
        ("Base checkpoint", "Casanovo 5.2.0 released weights (casanovo_orbitrap_v5-2-0.ckpt)", "Casanovo-TMT (final checkpoint)", "Starting weights for fine-tuning; weights used for inference"),
        ("Optimizer", "Adam (torch.optim.Adam)", "not used", "Optimizer with weight decay as listed above"),
        ("Training steps", f"{FINAL_STEP:,}", "not used", "Optimizer steps; about 10 passes over the training partition; the checkpoint at the last step was used"),
        ("Training spectra", "21,036,675", "not used", "Training partition as loaded: of its 21,688,803 spectra, the 652,128 with a precursor charge above 4 or fewer than 20 peaks were excluded"),
        ("Validation spectra", "2,631,393 (precursor charge ≤ 4)", "not used", "Validation partition as loaded (spectra with charge above 4 are dropped by the loader)"),
        ("Training spectrum order", "Steps 0–2,360,000: file-name order; steps 2,360,000–3,300,000: interleaved copy", "not used", "See Supplementary Fig. S1"),
        ("Hardware", "One NVIDIA GeForce RTX 5070 Ti", "One NVIDIA GeForce RTX 5070 Ti", "About 12.5 days of training computation"),
    ]
    for k, t, i, dsc in run:
        add("Run", k, t, i, dsc)
    save_table(pd.DataFrame(rows), "tableS3_configuration", "Table S3")


# ───────────────────────────── Table S3 Fig. 4D ─────────────────────────────
def tableS3():
    h = json.load(open(os.path.join(V, "harvest_gain.json"), encoding="utf-8"))
    rows = []
    for r in sorted(h["studies"], key=lambda r: int(DISPLAY[r["name"]].split()[1])):
        rows.append({"Study": DISPLAY[r["name"]],
                     "In training corpus": "no" if r["held_out"] else "yes",
                     "Phosphopeptides, SEQUEST HT": r["sequest"],
                     "Phosphopeptides, Casanovo-TMT": r["ours"],
                     "More than SEQUEST HT (%)": round(100 * (r["ours"] / r["sequest"] - 1), 1)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"  Spearman rho = {h['rho']}, exact two-sided P = {h['p']:.5f}")
    save_table(df, "tableS4_phosphopeptide_counts", "Table S4")


# ───────────────────────────── Table S4 검색 매개변수 ─────────────────────────────
def pd_observed():
    """PD 내보내기(PSMs.txt)에서 관측되는 값만 뽑는다 — 검색 설정 자체는 내보내기에 없다."""
    obs = {}
    for eng, suf in (("SEQUEST HT", "_PSMs.txt"), ("CHIMERYS", "-(1)_PSMs.txt")):
        mods, node, conf = collections.Counter(), collections.Counter(), collections.Counter()
        mc, ch, dm, ln, n = collections.Counter(), collections.Counter(), [], [], 0
        for tag, b in PD_BASES.items():
            with open(PD_DIR + b + suf, encoding="utf-8-sig", errors="replace") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    n += 1
                    node[row["Identifying Node"]] += 1; conf[row["Confidence"]] += 1
                    for m in re.findall(r"\(([^)]+)\)", row["Modifications"]):
                        if m != "Prot":
                            mods[m] += 1
                    mc[int(row["# Missed Cleavages"])] += 1; ch[int(row["Charge"])] += 1
                    try:
                        dm.append(float(row["DeltaM [ppm]"]))
                    except ValueError:
                        pass
                    ln.append(len(row["Sequence"]))
        dm = np.array(dm)
        obs[eng] = dict(n=n, node=dict(node), conf=dict(conf), mods=[m for m, _ in mods.most_common()],
                        mc=max(mc), ch=(min(ch), max(ch)), dm=(dm.min(), dm.max()), len=(min(ln), max(ln)))
        print(f"  PD {eng}: {n:,} PSMs, nodes {dict(node)}, confidence {dict(conf)}, mods {obs[eng]['mods']},"
              f" missed cleavages ≤{max(mc)}, charge {min(ch)}–{max(ch)}, ΔM {dm.min():.2f} to {dm.max():.2f} ppm,"
              f" length {min(ln)}–{max(ln)}")
    return obs


def fp_params(path):
    p = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = [x.strip() for x in line.split("=", 1)]
            p[k] = v
    return p


def fp_workflow(path):
    w = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        if "=" in line and not line.startswith("#"):
            k, v = line.rstrip("\n").split("=", 1)
            w[k] = v.replace("\\", "")
    return w


def fp_varmods(wf):
    out = []
    for it in wf["msfragger.table.var-mods"].split(";"):
        mass, site, on, mx = [x.strip() for x in it.split(",")]
        if on == "true":
            name = {"[^": "protein N terminus", "n^": "peptide N terminus"}.get(site, site)
            out.append(f"{name} {float(mass):+.4f} (max {mx})")
    return "; ".join(out)


def versions(log):
    t = open(log, encoding="utf-8", errors="replace").read()
    g = lambda pat: (re.search(pat, t) or [None, "?"])[1]
    return dict(fragpipe=g(r"FragPipe version ([\d.]+)"), msfragger=g(r"MSFragger version ([\d.]+)"),
                percolator=g(r"Percolator version ([\d.]+)"), philosopher=g(r"Philosopher version ([\d.]+)"))


def tableS4():
    pdo = pd_observed()
    # FragPipe (Studies 1–3): 세 스터디가 같은 workflow·params 를 썼는지 확인
    pars = {st: fp_params(f"{FP_RUN}/out_{st}/fragger.params") for st in FP_STUDY.values()}
    keys = ("precursor_mass_lower", "precursor_mass_upper", "fragment_mass_tolerance", "isotope_error",
            "variable_mod_01", "variable_mod_03", "add_K_lysine", "add_Nterm_peptide", "add_C_cysteine",
            "allowed_missed_cleavage_1", "digest_min_length", "digest_max_length", "digest_mass_range")
    for k in keys:
        assert len({pars[s].get(k) for s in pars}) == 1, k
    fp = pars["s2"]
    wf = fp_workflow(f"{FP_RUN}/phospho.workflow")
    ver = versions(f"{FP_RUN}/log_s2.txt")
    db = f"{FP_RUN}/db/2026-10-29-decoys-contam-target.fasta.fas"
    heads = [l for l in open(db, encoding="utf-8", errors="replace") if l.startswith(">")]
    n_rev = sum(h.startswith(">rev_") for h in heads)
    n_tgt_file = sum(1 for l in open(f"{FP_RUN}/db/target.fasta", encoding="utf-8", errors="replace") if l.startswith(">"))
    n_cont = len(heads) - 2 * n_rev
    print(f"  FragPipe DB: {len(heads):,} entries = {n_rev:,} decoys + {n_rev:,} targets"
          f" ({n_tgt_file:,} from target.fasta + {n_rev - n_tgt_file} contaminants)")
    # FragPipe (공개 데이터, Fig. 3G 와 Fig. S2)
    pub = {a: (fp_params(os.path.join(BASE, "fragpipe_in", a, "fragger.params")),
               fp_workflow(os.path.join(BASE, "fragpipe_in", a, "fragpipe.workflow")),
               versions(glob.glob(os.path.join(BASE, "fragpipe_in", a, "log_*.txt"))[0])) for a in PUBLIC}
    # Sage
    sg = {t: json.load(open(os.path.join(BASE, "out", "sage", t, "config.json"), encoding="utf-8")) for t, _, _ in STUDIES}
    for t in sg:
        a = {k: v for k, v in sg[t].items() if k not in ("mzml_paths", "output_directory")}
        b = {k: v for k, v in sg["study1"].items() if k not in ("mzml_paths", "output_directory")}
        assert a == b, t
    s = sg["study1"]; sd = s["database"]
    sfa = sd["fasta"]
    n_sage = sum(1 for l in open(sfa, encoding="utf-8", errors="replace") if l.startswith(">")) if os.path.exists(sfa) else 17229

    se, chm = pdo["SEQUEST HT"], pdo["CHIMERYS"]
    na = "not recorded in the exported results"
    p_fix = "TMTpro (+304.2071) on peptide N termini and K; carbamidomethyl (+57.0215) on C"
    rows = [
        ("Software", "Proteome Discoverer 3.3, SEQUEST HT node", "Proteome Discoverer 3.3, CHIMERYS node",
         f"FragPipe {ver['fragpipe']}, MSFragger {ver['msfragger']}, Percolator {ver['percolator']}, Philosopher {ver['philosopher']}",
         "Sage 0.14.6",
         f"FragPipe {pub['PXD063976'][2]['fragpipe']}, MSFragger {pub['PXD063976'][2]['msfragger']}, Percolator {pub['PXD063976'][2]['percolator']}, Philosopher {pub['PXD063976'][2]['philosopher']}"),
        ("Used for", "Studies 1–3 (Figs. 4 and 5); reference for peptidoform precision on Studies 1–3",
         "Studies 1–3 (Figs. 4 and 5)", "Studies 1–3 (Figs. 4 and 5; Supplementary Fig. S3)", "Studies 1–3 (Fig. 5B)",
         "PXD054085, PXD072211, PXD063976 (Fig. 3G; Supplementary Fig. S2)"),
        ("Workflow", "—", "—", "Built-in TMT16-phospho workflow; quantification modules off", "—",
         "Built-in basic TMT16 workflow (PXD063976: TMT16-phospho)"),
        ("Protein database",
         "Swiss-Prot mouse reference proteome (17,229 entries) + 4,604 unreviewed entries for genes absent from it",
         "Same as SEQUEST HT",
         f"Same {n_tgt_file:,} entries + {n_rev - n_tgt_file} common contaminants; reversed decoys (rev_) added by Philosopher",
         f"Swiss-Prot mouse reference proteome ({n_sage:,} entries, UniProtKB 2025_01); reversed decoys generated by Sage",
         "UniProt reviewed reference proteome (mouse UP000000589; human UP000005640) + contaminants + reversed decoys"),
        ("Enzyme", na + "; observed missed cleavages ≤ %d" % se["mc"], na + "; observed missed cleavages ≤ %d" % chm["mc"],
         f"Strict trypsin (cleaves after K/R, including before P), fully specific, up to {fp['allowed_missed_cleavage_1']} missed cleavages",
         f"Trypsin (cleaves after K/R, not before P), up to {sd['enzyme']['missed_cleavages']} missed cleavages",
         f"Strict trypsin, fully specific, up to {pub['PXD063976'][0]['allowed_missed_cleavage_1']} missed cleavages"),
        ("Peptide length and mass", na + "; observed length %d–%d" % se["len"], na + "; observed length %d–%d" % chm["len"],
         f"{fp['digest_min_length']}–{fp['digest_max_length']} residues; {fp['digest_mass_range'].replace(' ', '–')} Da",
         f"{sd['enzyme']['min_len']}–{sd['enzyme']['max_len']} residues; {sd['peptide_min_mass']:g}–{sd['peptide_max_mass']:g} Da",
         f"{pub['PXD063976'][0]['digest_min_length']}–{pub['PXD063976'][0]['digest_max_length']} residues; "
         f"{pub['PXD063976'][0]['digest_mass_range'].replace(' ', '–')} Da"),
        ("Precursor tolerance", na + "; observed mass error %.1f to +%.1f ppm" % se["dm"],
         na + " (mass error reported as 0 for every match)",
         f"{fp['precursor_mass_lower']} to +{fp['precursor_mass_upper']} ppm; isotope errors {fp['isotope_error'].replace('/', ', ')}; mass calibration on",
         f"{s['precursor_tol']['ppm'][0]} to +{s['precursor_tol']['ppm'][1]} ppm; isotope errors {', '.join(map(str, s['isotope_errors']))}",
         f"{pub['PXD063976'][0]['precursor_mass_lower']} to +{pub['PXD063976'][0]['precursor_mass_upper']} ppm; isotope errors {pub['PXD063976'][0]['isotope_error'].replace('/', ', ')}"),
        ("Fragment tolerance", na, na, f"{fp['fragment_mass_tolerance']} ppm; b and y ions, fragment charge ≤ {fp['max_fragment_charge']}",
         f"{s['fragment_tol']['ppm'][0]} to +{s['fragment_tol']['ppm'][1]} ppm; b and y ions, fragment charge ≤ {s['max_fragment_charge']}",
         f"{pub['PXD063976'][0]['fragment_mass_tolerance']} ppm"),
        ("Fixed modifications", na + "; observed: " + ", ".join(m for m in se["mods"] if m in ("TMTpro", "Carbamidomethyl")),
         na + "; observed: " + ", ".join(m for m in chm["mods"] if m in ("TMTpro", "Carbamidomethyl")),
         p_fix, "TMTpro (+304.2071) on peptide N termini and K; carbamidomethyl (+57.0215) on C",
         "Carbamidomethyl on C; TMTpro on K; TMTpro on peptide N termini (PXD063976 only)"),
        ("Variable modifications",
         na + "; observed: " + ", ".join(m for m in se["mods"] if m not in ("TMTpro", "Carbamidomethyl")),
         na + "; observed: " + ", ".join(m for m in chm["mods"] if m not in ("TMTpro", "Carbamidomethyl")),
         fp_varmods(wf) + f"; at most {fp['max_variable_mods_per_peptide']} per peptide",
         f"Phospho (+79.9663) on S, T, Y; oxidation (+15.9949) on M; at most {sd['max_variable_mods']} per peptide",
         "PXD063976: " + fp_varmods(pub["PXD063976"][1]) + "; PXD054085 and PXD072211: " + fp_varmods(pub["PXD054085"][1])),
        ("Rescoring and FDR",
         "q-value < 0.01 (high confidence) at the PSM level, as exported; q-value, PEP and SVM score columns present",
         "q-value < 0.01 (high confidence) at the PSM level, as exported",
         "Percolator rescoring; PTMProphet site localization; Philosopher filter at 1% PSM, peptide and protein FDR (sequential, picked protein FDR)",
         "Sage built-in rescoring (linear discriminant analysis, default); 1% spectrum-level FDR (Fig. 5B)",
         "Percolator rescoring; Philosopher filter at 1% PSM, peptide and protein FDR"),
        ("Matches in the exports (Studies 1–3)", f"{se['n']:,} PSMs", f"{chm['n']:,} PSMs", "—", "—", "—"),
    ]
    cols = ["Parameter", "SEQUEST HT", "CHIMERYS", "FragPipe 22 (Studies 1–3)", "Sage 0.14.6 (Studies 1–3)",
            "FragPipe 24 (public datasets)"]
    df = pd.DataFrame(rows, columns=cols)
    save_table(df, "tableS2_search_parameters", "Table S2")


if __name__ == "__main__":
    which = [w.upper() for w in sys.argv[1:]] or ["S1", "S2", "S3", "T2", "T3", "T4"]
    for w, f in (("T2", tableS2), ("T3", tableS3), ("T4", tableS4), ("S1", figS1), ("S2", figS2), ("S3", figS3)):
        if w in which:
            print(f"── {w}")
            f()
