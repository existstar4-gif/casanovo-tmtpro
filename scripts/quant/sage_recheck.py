# -*- coding: utf-8 -*-
"""sage_recheck.py — 현행 기준(인산화 그룹 FDR, N말단 미표지 제외)의 검증 집단에서 Sage 일치를 다시 잰다 (2026-11-03).

입력: export_validation_groups.py 의 groups_<study>.tsv (펩타이드마다 최고 점수 인산화 PSM), out/sage/<study>/results.sage.tsv
스캔 단위로 Sage 1등과 비교: Sage 확신(spectrum_q ≤ 0.01, target) 중 같은 bare 서열 비율 (+ 인산 자리까지 같은 비율).
Sage 설정(TMTpro N말단·K 고정, 가변 수식 2개 이하)으로 찾을 수 없는 예측은 따로 센다.

    python scripts/quant/sage_recheck.py
"""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_sage import bare, phospho_sites, searchable, read_sage   # noqa: E402

res = {}
for st in ["study1", "study2", "study3"]:
    sage = read_sage(f"out/sage/{st}/results.sage.tsv")
    c = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(f"out/quant/validation/groups_{st}.tsv", encoding="utf-8"), delimiter="\t"):
        g = r["group"]
        if g.startswith("wrong") and r["above_cut"] != "1":
            continue
        ok, _ = searchable(r["pf"])
        if not ok:
            c[g]["unsearchable"] += 1; continue
        s = sage.get((r["stem"], int(r["scan"])))
        if s is None or s.get("label") == "-1":
            c[g]["sage_none_or_decoy"] += 1; continue
        conf = float(s["spectrum_q"]) <= 0.01
        same = bare(s["peptide"]) == r["bare"].replace("I", "L")
        site = same and phospho_sites(s["peptide"]) == phospho_sites(r["pf"])
        key = "conf" if conf else "weak"
        c[g][key] += 1; c[g][key + "_same"] += same; c[g][key + "_site"] += site
    out = {}
    for g, d in c.items():
        out[g] = dict(d)
        for key in ("conf", "weak"):
            if d[key]:
                out[g][key + "_same_frac"] = round(d[key + "_same"] / d[key], 3)
                out[g][key + "_site_frac_of_same"] = round(d[key + "_site"] / max(d[key + "_same"], 1), 3)
    res[st] = out
    print(st, json.dumps(out, ensure_ascii=False))
json.dump(res, open("out/quant/validation/sage_recheck.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
