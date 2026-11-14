# -*- coding: utf-8 -*-
"""compose_fig5.py — Figure 5 = make_figures.fig4() 가 그린 검증 2x2 (A–D, make_figures_phospho.fig8) 를 복사한다 — E–F(구제율)는 2026-11-13 삭제 (2026-11-13).

사용: python scripts/make_figures.py 를 먼저 돌린 뒤 실행.
"""
import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(BASE, "figures")
for ext in ("png", "pdf"):
    shutil.copy(os.path.join(F, f"fig8_ours_only_validation.{ext}"), os.path.join(F, "manuscript_v3", f"Figure5.{ext}"))
print("Figure5.png / .pdf <- fig8_ours_only_validation")
