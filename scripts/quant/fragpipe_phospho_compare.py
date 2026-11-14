# -*- coding: utf-8 -*-
"""fragpipe_phospho_compare.py — FragPipe 22 결과를 PD·우리와 같은 규칙으로 센다 (2026-10-29, 사용자 요청).

FragPipe 실행: data/fragpipe/run_all.sh (TMT16-phospho, 정량 끔, PD 와 같은 합본 DB + rev_ decoy, PD 와 같은 분획).
세 가지 수를 낸다 (인산화 펩타이드 = bare 서열, I=L, 인산화 PSM 이 하나라도 있으면 인산화):
  A  FragPipe 기본     = philosopher 출력 psm.tsv (PSM 1% + 펩타이드 1% + 단백질 1% 순차 필터)
  B  공동 FDR 1%       = Percolator 전체 PSM 표(target + decoy)에서 펩타이드 단위로 다시 건 1% (단백질 필터 없음 = PD 와 같은 단위)
  C  인산화 그룹 FDR 1% = B 와 같되 인산화 PSM 만으로 문턱을 잡는다 (우리 분석법과 같은 규칙)
점수 = 1 − PEP (Percolator 사후 오류확률; 분획마다 따로 학습한 SVM 점수보다 분획 사이에 비교 가능하다. philosopher 도 이 확률을 쓴다).
정량 가능 = quantifiable_share_rule 과 같은 새 규칙 (TMT 표지 · 자기 몫 ≥ 0.5 · 채널 ≥ 8), 리포터는 우리 *.rep.tsv(스캔 단위).

    python scripts/quant/fragpipe_phospho_compare.py --study s1
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantifiable_share_rule as qr          # noqa: E402
import quantifiable_same_rule as qs           # noqa: E402
import chimeric_case_study as cs              # noqa: E402

FP = "data/fragpipe"
TAG = {"s1": "study1", "s2": "study2", "s3": "study3"}
MODS = {"79.9663": "Phospho", "15.9949": "Oxidation", "304.2072": "TMTpro", "304.2071": "TMTpro", "42.0106": "Acetyl"}


def parse_pin_peptide(p):
    """'K.n[304.2072]FLES[79.9663]K3.A' -> (seq, {pos: mod}) ; K 의 TMTpro 와 C 의 카바미도메틸은 고정이라 표기가 없다."""
    core = p.split(".", 1)[1].rsplit(".", 1)[0]
    core = re.sub(r"\d+$", "", core)                      # 끝에 붙은 전하 숫자
    seq, mods = "", {}
    for tok, mass in re.findall(r"([A-Zn])(?:\[([\d.]+)\])?", core):
        if tok == "n":
            if mass:
                mods[0] = MODS.get(mass, mass)
            continue
        seq += tok
        if mass:
            mods[len(seq)] = MODS.get(mass, mass)
    for i, a in enumerate(seq):
        if a == "K":
            mods[i + 1] = "TMTpro"
    return seq, mods


def read_percolator(st):
    """[dict(stem, scan, z, bare, phos, prob, decoy, mz, lab)] — rank 1 만"""
    out = []
    for kind in ("target", "decoy"):
        for p in sorted(glob.glob(f"{FP}/out_{st}/{st}/*_percolator_{kind}_psms.tsv")):
            for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
                pid, rank = r["PSMId"].rsplit("_", 1)
                if rank != "1":
                    continue
                stem, scan, _s2, z = pid.rsplit(".", 3)
                seq, mods = parse_pin_peptide(r["peptide"])
                if any(a not in cs.AA for a in seq):          # U(셀레노시스테인) 등 비표준 잔기 — 몇 개뿐이라 뺀다
                    continue
                z = int(z)
                out.append(dict(stem=stem, scan=int(scan), z=z, bare=seq.replace("I", "L"), phos="Phospho" in mods.values(),
                                prob=1.0 - float(r["posterior_error_prob"]), decoy=kind == "decoy",
                                mz=(cs.mass_of(seq, mods) + z * cs.PROTON) / z, lab=qr.labelled(seq, mods)))
    return out


def cut_for(best, fdr=0.01):
    """best = {bare: (score, decoy)} -> 문턱 (누적 decoy/target ≤ fdr 인 가장 낮은 점수)"""
    nt = nd = 0
    cut = None
    for sc, dec in sorted(best.values(), key=lambda x: -x[0]):
        nd += dec; nt += not dec
        if nt and nd / nt <= fdr:
            cut = sc
    return cut


def read_default(st):
    """philosopher psm.tsv -> PSM 목록 (target 만, 이미 필터됨)"""
    out = []
    for r in csv.DictReader(open(f"{FP}/out_{st}/{st}/psm.tsv", encoding="utf-8"), delimiter="\t"):
        stem, scan, _s2, z = r["Spectrum"].rsplit(".", 3)
        am = r["Assigned Modifications"]
        seq = r["Peptide"]
        nk = sum(1 for m in re.findall(r"(\d+)K\(304\.20", am))
        out.append(dict(stem=stem, scan=int(scan), z=int(z), bare=seq.replace("I", "L"), phos="(79.9663)" in am,
                        mz=float(r["Calculated M/Z"]), lab=("N-term(304.20" in am and nk == seq.count("K"))))
    return out


def phospho_bare(psms):
    seen_p = {p["bare"] for p in psms if p["phos"]}
    allb = {p["bare"] for p in psms}
    return seen_p, allb - seen_p


def quantifiable(st, sets):
    rep = qs.read_rep(sorted(glob.glob(f"out/quant/{TAG[st]}/*.rep.tsv")))
    want = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, ps in sets.items():
        for i, p in enumerate(ps):
            p["id"] = (name, i)
            want[p["stem"]][p["scan"]].append((p["id"], p["mz"], p["z"]))
    share = {}
    for k, stem in enumerate(sorted(want), 1):
        share.update(qr.own_shares(stem, want[stem]))
        print(f"  [{k}/{len(want)}] {stem}", flush=True)
    res = {}
    for name, ps in sets.items():
        q, why = set(), collections.Counter()
        for p in ps:
            v = rep.get((p["stem"], p["scan"]))
            if v is None:
                why["no reporter row"] += 1; continue
            s = share.get(p["id"], float("nan"))
            if not p["lab"]:
                why["not fully TMT-labelled"] += 1
            elif not (s >= 0.5):
                why["own share < 0.5"] += 1
            elif v[1] < 8:
                why["< 8 channels"] += 1
            else:
                q.add(p["bare"])
        res[name] = dict(quantifiable=len(q), psm_fail_reasons=dict(why))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True, choices=list(TAG))
    a = ap.parse_args()
    st = a.study
    res = {}
    # A. FragPipe 기본
    dflt = read_default(st)
    ph, no = phospho_bare(dflt)
    res["A_default"] = dict(phospho=len(ph), nonphospho=len(no), psms=len(dflt))
    # B / C. Percolator 전체 표에서 다시 건 FDR
    perc = read_percolator(st)
    best, best_p = {}, {}
    for p in perc:
        if p["prob"] > best.get(p["bare"], (-1, 0))[0]:
            best[p["bare"]] = (p["prob"], p["decoy"])
        if p["phos"] and p["prob"] > best_p.get(p["bare"], (-1, 0))[0]:
            best_p[p["bare"]] = (p["prob"], p["decoy"])
    phos_seen = set(best_p)
    cj = cut_for(best)
    cp = cut_for(best_p)
    bn = {k: v for k, v in best.items() if k not in phos_seen}
    cn = cut_for(bn)
    j_ph = {k for k, v in best.items() if v[0] >= cj and not v[1] and k in phos_seen}
    j_dec = sum(1 for v in best.values() if v[0] >= cj and v[1])
    g_ph = {k for k, v in best_p.items() if v[0] >= cp and not v[1]}
    g_dp = sum(1 for v in best_p.values() if v[0] >= cp and v[1])
    g_no = {k for k, v in bn.items() if v[0] >= cn and not v[1]}
    g_dn = sum(1 for v in bn.values() if v[0] >= cn and v[1])
    res["B_joint_1pct"] = dict(cut_prob=cj, phospho=len(j_ph), decoy_peptides_all=j_dec)
    res["C_phospho_group_1pct"] = dict(cut_prob=cp, phospho=len(g_ph), phospho_decoys=g_dp, nonphospho_cut=cn, nonphospho=len(g_no),
                                       combined_fdr=round((g_dp + g_dn) / max(1, len(g_ph) + len(g_no)), 4))
    # 정량 가능 (인산화 PSM 만)
    sets = {"A_default": [p for p in dflt if p["phos"]],
            "B_joint_1pct": [p for p in perc if p["phos"] and not p["decoy"] and p["prob"] >= cj],
            "C_phospho_group_1pct": [p for p in perc if p["phos"] and not p["decoy"] and p["prob"] >= cp]}
    for k, v in quantifiable(st, sets).items():
        res[k].update(v)
    os.makedirs("out/fragpipe_heldout", exist_ok=True)
    json.dump(res, open(f"out/fragpipe_heldout/{st}.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
