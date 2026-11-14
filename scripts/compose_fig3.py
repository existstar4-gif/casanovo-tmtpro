# -*- coding: utf-8 -*-
"""compose_fig3.py — Figure 3 = 외부 데이터(A-E, fig3_scope) 위 + re-ranking(F-H, fig2b_rerank) 아래.

2026-11-11 사용자 결정으로 두 그림을 합쳤다. 두 PNG 는 모두 300 dpi 라 글자 크기가 같다.
아랫줄은 크기를 바꾸지 않고 왼쪽 정렬로 붙인다(늘리면 글자 크기가 달라진다).
사용: python scripts/make_figures.py 의 fig3() 와 scripts/make_rerank_fig.py 를 먼저 돌린 뒤 실행.
"""
import os

from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(BASE, "figures")
top = Image.open(os.path.join(F, "fig3_scope.png")).convert("RGB")
bot_f = Image.open(os.path.join(F, "fig2b_rerank.png")).convert("RGB")
# 2026-11-12: 옛 Fig. 4 (A-C) 를 Fig. 3 G-I 로 합쳤다 — 아랫줄 F 옆에 붙인다(크기 그대로).
# 2026-11-12: H·I 는 Fig. 6 으로 뺐다 — 이제 G 하나만 붙인다.
ghi = Image.open(os.path.join(F, "fig3_g.png")).convert("RGB")
bot = Image.new("RGB", (bot_f.width + ghi.width, max(bot_f.height, ghi.height)), "white")
bot.paste(bot_f, (0, 0))
bot.paste(ghi, (bot_f.width, 0))
gap = 60
w = max(top.width, bot.width)
out = Image.new("RGB", (w, top.height + gap + bot.height), "white")
out.paste(top, (0, 0))
out.paste(bot, (0, top.height + gap))
for p in (os.path.join(F, "fig3_merged.png"), os.path.join(F, "manuscript_v3", "Figure3.png")):
    out.save(p, dpi=(300, 300))
# PIL 의 PDF 저장은 이 환경에 JPEG 인코더가 없어 실패한다 — matplotlib 로 같은 픽셀을 싣는다.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(out.width / 300, out.height / 300), dpi=300)
fig.figimage(np.asarray(out), 0, 0)
fig.savefig(os.path.join(F, "manuscript_v3", "Figure3.pdf"), dpi=300)
plt.close(fig)
print("Figure3.png / .pdf", out.size)
