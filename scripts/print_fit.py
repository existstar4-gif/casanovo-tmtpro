# -*- coding: utf-8 -*-
"""print_fit.py — 그림을 인쇄 폭에 맞춰 저장한다 (2026-11-18, 김광은 교수님 C26 "Figure 글자가 작다").

원고는 그림을 Word 페이지 폭(6.5 in)으로 줄여 넣는다. 그림을 13–20 in 폭으로 그려 두면
9 pt 글자가 인쇄에서 3–5 pt 가 된다. 여기서는
  ① 그림이 실제로 인쇄될 폭(print_w, inch)을 정하고
  ② 그 폭으로 줄였을 때 작은 글자(하위 10%)가 min_pt 이상이 되도록 모든 글자를 같은 배율로 키운 뒤
  ③ 레이아웃을 다시 잡고, 픽셀 폭이 print_w × 600 dpi 가 되도록 저장한다.
글자 사이의 크기 비율(제목 > 축 이름 > 눈금)은 그대로 유지된다.
"""
from __future__ import annotations

import os

import matplotlib.text as mtext
import numpy as np

PRINT_DPI = 600


def _texts(fig):
    out = []
    for t in fig.findobj(mtext.Text):
        if t.get_visible() and t.get_text().strip() and t.figure is not None:
            out.append(t)
    return out


def _bbox_width(fig):
    r = fig.canvas.get_renderer()
    return fig.get_tightbbox(r).width          # inch


def printed_sizes(fig, print_w):
    """지금 상태로 print_w 폭에 인쇄하면 각 글자가 몇 pt 가 되는지."""
    k = print_w / _bbox_width(fig)
    return np.array([t.get_fontsize() * k for t in _texts(fig)])


def fit(fig, print_w=6.5, min_pt=7.0, relayout=True, rounds=4):
    """작은 글자(하위 10%)가 인쇄에서 min_pt 가 되도록 글자를 키운다. 이미 크면 그대로 둔다."""
    for _ in range(rounds):
        sizes = printed_sizes(fig, print_w)
        if not len(sizes):
            return 1.0
        low = float(np.percentile(sizes, 10))
        if low >= min_pt * 0.98:
            break
        k = min_pt / low
        for t in _texts(fig):
            t.set_fontsize(t.get_fontsize() * k)
        if relayout:
            try:
                if fig.get_layout_engine() is None:
                    fig.tight_layout()
            except Exception:                    # noqa: BLE001 — 수동 배치 그림은 그대로 둔다
                pass
    return float(np.percentile(printed_sizes(fig, print_w), 10))


def save(fig, out_dir, name, print_w=6.5, min_pt=7.0, exts=("png", "pdf"), relayout=True):
    low = fit(fig, print_w, min_pt, relayout)
    bw = _bbox_width(fig)
    dpi = PRINT_DPI * print_w / bw              # 저장 픽셀 폭 = print_w × 600
    paths = []
    for ext in exts:
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    sizes = printed_sizes(fig, print_w)
    print(f"  {name}: print width {print_w} in, printed text {sizes.min():.1f}–{sizes.max():.1f} pt "
          f"(10th pct {low:.1f})")
    return paths
