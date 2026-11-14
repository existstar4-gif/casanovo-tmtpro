# -*- coding: utf-8 -*-
"""same_site_all_methods.py — 같은 인산화 자리 짝과의 리포터 상관을 네 방법 모두에 같은 규칙으로 잰다 (2026-11-13, 사용자 질문).

질문: Fig. 5D 의 'Casanovo-TMT 만 찾은 인산화 펩타이드 ↔ 같은 자리 짝' 상관(0.27/0.48/0.37)이
      다른 검색 엔진이 '혼자만 찾은' 인산화 펩타이드에서는 어떻게 나오는가.

규칙 (네 방법 공통):
  집합      Fig. 4A 와 같은 기준의 인산화 bare 서열 (overlap4_phospho 의 SEQUEST HT · CHIMERYS · FragPipe · Casanovo-TMT)
  유일      한 방법만 찾은 서열 (Fig. 4C 의 'unique')
  합의      두 방법 이상이 찾은 서열 — 짝의 출처이자 기준선
  리포터    스캔 단위 *.rep.tsv (네 방법이 같은 추출값을 쓴다). 펩타이드마다 리포터 합이 가장 큰 PSM 하나.
            log2 → 채널별 중앙값(적재량) 보정 → 행 중심화.
  자리      (대표 단백질, 단백질 안 인산화 위치 집합). 짝 = 같은 자리를 가진 다른 합의 서열.
  상관      대상 패턴 ↔ 짝 패턴들의 중앙값, Pearson.

    python scripts/quant/same_site_all_methods.py
출력: out/quant/validation/same_site_all_methods.json
"""
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overlap4_phospho as ov4                 # noqa: E402
import quantifiable_share_rule as qr           # noqa: E402
import fragpipe_phospho_compare as fpc         # noqa: E402
import count_matched_fdr as cm                 # noqa: E402
import chimeric_case_study as cs               # noqa: E402
import ours_only_validation as oov             # noqa: E402
from rescued_pick import FA, P, BASES          # noqa: E402

PDB = "data/pd/"
FP_ST = {"study1": "s1", "study2": "s2", "study3": "s3"}
METHODS = ["SEQUEST HT", "CHIMERYS", "FragPipe", "Casanovo-TMT"]


def read_rep(study):
    rep, hdr = {}, None
    for p in sorted(glob.glob(f"out/quant/{study}/*.rep.tsv")):
        for line in open(p, encoding="utf-8"):
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[0] == "stem":
                hdr = c; i0 = hdr.index("ms1_scan") + 1
                continue
            rep[(c[0], int(c[1]))] = np.array([float(x) if x else 0.0 for x in c[i0:]], float)
    return rep


def phos_pos(seq, mods):
    """1-based 잔기 위치 중 인산화된 것"""
    return tuple(sorted(k for k, v in mods.items() if v == "Phospho" and k >= 1))


def psms_pd(path):
    out = []
    for p in qr.read_pd(path):
        out.append((p["stem"], p["scan"], p["bare"], phos_pos(p["_seq"], p["_mods"])))
    return out


def psms_fp(st):
    out = []
    for r in csv.DictReader(open(f"{fpc.FP}/out_{st}/{st}/psm.tsv", encoding="utf-8"), delimiter="\t"):
        am = r["Assigned Modifications"]
        if "(79.9663)" not in am:
            continue
        stem, scan, _s2, _z = r["Spectrum"].rsplit(".", 3)
        pos = tuple(sorted(int(m) for m in re.findall(r"(\d+)[STY]\(79\.96", am)))
        out.append((stem, int(scan), r["Peptide"].replace("I", "L"), pos))
    return out


def psms_ours(study, cut):
    out = []
    for p in oov.read_psms(study):           # N말단 표지(또는 아세틸) 판독만
        if "Phospho" not in p["pf"] or p["score"] < cut:
            continue
        seq, mods = cs.our_form(p["pf"])
        out.append((p["stem"], p["scan"], p["bare"], phos_pos(seq, mods)))
    return out


def main():
    res = {}
    mouse = oov.proteins(FA)
    mouse_n = {a: cm.normalize(s) for a, s in mouse.items()}
    for study in ("study1", "study2", "study3"):
        b = BASES[study]
        sets = {"SEQUEST HT": ov4.cps.read_pd(P + b + "_PeptideGroups.txt")[0],
                "CHIMERYS": ov4.cps.read_pd(P + b + "-(1)_PeptideGroups.txt")[0],
                "FragPipe": ov4.fragpipe(study), "Casanovo-TMT": ov4.ours(study)}
        import re as _re
        cut = float(_re.search(r"인산화 문턱 = ([0-9.]+)", open(f"out/quant/split_phospho_pfdr_lab_{study}.log", encoding="utf-8").read()).group(1))
        psms = {"SEQUEST HT": psms_pd(PDB + b + "_PSMs.txt"), "CHIMERYS": psms_pd(PDB + b + "-(1)_PSMs.txt"),
                "FragPipe": psms_fp(FP_ST[study]), "Casanovo-TMT": psms_ours(study, cut)}
        rep = read_rep(study)
        allv = np.array(list(rep.values()))
        active = np.where(np.mean(allv > 0, axis=0) > 0.2)[0]   # 대부분 스캔은 리포터 0 — 쓰이는 채널만
        logv = np.log2(np.where(allv[:, active] > 0, allv[:, active], np.nan))
        load = np.nanmedian(logv - np.nanmean(logv, axis=1, keepdims=True), axis=0)

        def prof(v):
            c = v[active]
            if np.any(c <= 0):
                return None
            x = np.log2(c); x = x - x.mean() - load
            return x - x.mean()

        tgt, _ = cm.target_decoy(FA, set().union(*sets.values()))

        def site_key(bare, pos):
            for acc in sorted(tgt.get(bare, ())):
                i = mouse_n.get(acc, "").find(bare)
                if i >= 0 and pos:
                    return (acc, frozenset(i + x for x in pos))
            return None

        # 방법·펩타이드마다 대표 PSM (리포터 합 최대)
        rep_of = {}
        for m, ps in psms.items():
            best = {}
            for stem, scan, bare, pos in ps:
                if bare not in sets[m]:
                    continue
                v = rep.get((stem, scan))
                if v is None:
                    continue
                s = float(v.sum())
                if s > best.get(bare, (-1,))[0]:
                    best[bare] = (s, v, pos)
            rep_of[m] = best
        count = collections.Counter(k for m in METHODS for k in sets[m])
        consensus = {k for k, n in count.items() if n >= 2}
        # 합의 서열의 자리 → 패턴 (여러 방법 중 리포터 합 최대 PSM)
        cons_prof = {}
        for k in consensus:
            cand = [rep_of[m][k] for m in METHODS if k in rep_of[m]]
            if not cand:
                continue
            s, v, pos = max(cand, key=lambda t: t[0])
            pv, sk = prof(v), site_key(k, pos)
            if pv is not None and sk:
                cons_prof[k] = (sk, pv)
        by_site = collections.defaultdict(list)
        for k, (sk, pv) in cons_prof.items():
            by_site[sk].append((k, pv))

        def corr_for(items):
            rs = []
            for k, v, pos in items:
                pv, sk = prof(v), site_key(k, pos)
                if pv is None or not sk:
                    continue
                mates = [mv for kk, mv in by_site.get(sk, []) if kk != k]
                if mates:
                    r = float(np.corrcoef(pv, np.median(np.array(mates), axis=0))[0, 1])
                    if np.isfinite(r):
                        rs.append(r)
            return rs

        out = {}
        for m in METHODS:
            uniq = [k for k in sets[m] if count[k] == 1]
            items = [(k,) + rep_of[m][k][1:] for k in uniq if k in rep_of[m]]
            rs = corr_for(items)
            out[m] = dict(unique=len(uniq), with_partner=len(rs), frac_with_partner=round(len(rs) / max(1, len(uniq)), 3),
                          mean_r=round(float(np.mean(rs)), 3) if rs else None)
        items = [(k,) + max([rep_of[m][k] for m in METHODS if k in rep_of[m]], key=lambda t: t[0])[1:]
                 for k in consensus if any(k in rep_of[m] for m in METHODS)]
        rs = corr_for(items)
        out["consensus (>=2 methods)"] = dict(n=len(consensus), with_partner=len(rs), mean_r=round(float(np.mean(rs)), 3) if rs else None)
        res[study] = out
        print(study, json.dumps(out, ensure_ascii=False), flush=True)
    json.dump(res, open("out/quant/validation/same_site_all_methods.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
