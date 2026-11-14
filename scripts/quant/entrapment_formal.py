# -*- coding: utf-8 -*-
"""entrapment_formal.py — Wen et al. 2025 (Nat Methods 22:1454) 형식의 entrapment 오류율 (2026-11-03).

두 종류의 entrapment:
  1. 외래종: 애기장대 Swiss-Prot 트립신 소화물 − 생쥐 target·decoy (r ≈ 0.667) — 무작위 오판독을 잰다
  2. 이웃 잔기 맞바꾸기: 생쥐 target 펩타이드마다 C말단을 뺀 한 자리에서 이웃한 두 잔기를 바꾼 서열 1개 (r ≈ 1, 짝 있음)
     — 질량이 같고 조각 대부분을 공유해 '순서만 틀린 그럴듯한 오판독'을 잰다 (OmniNovo 2025 프리프린트 방식)

추정식 (N_T = 문턱 위 target 펩타이드, N_E = 문턱 위 entrapment 펩타이드):
  하한     FDP_lb   = N_E / (N_T + N_E)
  결합     FDP_comb = N_E (1 + 1/r) / (N_T + N_E)
  짝(2만)  FDP_pair = (N_E + N_{E≥s>T} + 2 N_{E>T≥s}) / (N_T + N_E)
문턱은 운영 문턱(생쥐 target·decoy 로 잡은 인산화 그룹 1%)을 그대로 쓴다. 집합: 전체 인산화, 우리만 전체·확신·6·구제.

    python scripts/quant/entrapment_formal.py --study study3
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import count_matched_fdr as cm                # noqa: E402
import compare_phospho_split as cps           # noqa: E402
import fragpipe_phospho_compare as fpc        # noqa: E402
import ours_only_validation as ov             # noqa: E402
from rescued_pick import FA, P, BASES         # noqa: E402


def swap_entrapment(targets, forbid, seed=0):
    """target → 짝 entrapment. 이웃한 서로 다른 두 잔기를 바꾼다(C말단 잔기 고정, I/L 은 같은 것으로 본다)."""
    rng = random.Random(seed)
    pair = {}
    for t in sorted(targets):
        pos = [i for i in range(len(t) - 2) if t[i] != t[i + 1]]
        rng.shuffle(pos)
        for i in pos:
            e = t[:i] + t[i + 1] + t[i] + t[i + 2:]
            if e not in forbid:
                pair[t] = e
                break
    return pair


def estimators(n_t, n_e, r, n_e_only=None, n_e_win=None):
    d = max(n_t + n_e, 1)
    out = dict(N_T=n_t, N_E=n_e, r=round(r, 4), fdp_lower=round(n_e / d, 4), fdp_combined=round(n_e * (1 + 1 / r) / d, 4))
    if n_e_only is not None:
        out["N_E_ge_s_gt_T"] = n_e_only
        out["N_E_gt_T_ge_s"] = n_e_win
        out["fdp_paired"] = round((n_e + n_e_only + 2 * n_e_win) / d, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="study3")
    a = ap.parse_args()
    psms = ov.read_psms(a.study)
    best, best_p = {}, {}
    for p in psms:
        k = p["bare"]
        if p["score"] > best.get(k, (-1,))[0]:
            best[k] = (p["score"], p)
        if "Phospho" in p["pf"] and p["score"] > best_p.get(k, (-1,))[0]:
            best_p[k] = (p["score"], p)
    tgt, dec = cm.target_decoy(FA, set(best))
    joint, _ = cm.cutoff_for_fdr({k: v[0] for k, v in best.items()}, tgt, dec, 0.01)
    cut_p, _ = cm.cutoff_for_fdr({k: v[0] for k, v in best_p.items()}, tgt, dec, 0.01)

    mouse = ov.proteins(FA)
    t_all = ov.full_digest(mouse.values()); d_all = ov.full_digest(mouse.values(), rev=True)
    e_all = ov.full_digest(ov.proteins(ov.ENT).values()) - t_all - d_all
    r_sp = len(e_all) / len(t_all)
    pair = swap_entrapment(t_all, t_all | d_all)
    sw_all = set(pair.values())
    r_sw = len(sw_all) / len(t_all)
    print(f"target {len(t_all):,}  arath-entrap {len(e_all):,} (r={r_sp:.3f})  swap-entrap {len(sw_all):,} (r={r_sw:.3f})", flush=True)

    other = ov.engine_phospho(a.study)   # 2026-11-13: Fig. 4C 기준 (엔진의 인산화 동정)

    def band(k, s):
        return "ours_conf"

    sc = {k: v[0] for k, v in best_p.items()}
    passed = {k for k, s in sc.items() if s >= cut_p}
    sets = collections.defaultdict(lambda: dict(T=set(), E_sp=set(), E_sw=set()))
    for k in passed:
        s = sc[k]
        names = ["all_phospho"]
        if k in tgt:
            if k not in other:
                names += ["ours_only", band(k, s)]
            for n in names:
                sets[n]["T"].add(k)
        else:
            # entrapment 판독은 DB 도구가 찾을 수 없으므로(생쥐 DB) 우리만 집합에도 들어간다
            names += ["ours_only", band(k, s)]
            for n in names:
                if k in e_all:
                    sets[n]["E_sp"].add(k)
                if k in sw_all:
                    sets[n]["E_sw"].add(k)
    res = dict(study=a.study, phospho_cut=cut_p, joint_cut=joint, r_arabidopsis=round(r_sp, 4), r_swap=round(r_sw, 4))
    inv = {e: t for t, e in pair.items()}
    for n in ["all_phospho", "ours_only", "ours_conf", "ours_resc"]:
        S = sets[n]
        n_t = len(S["T"])
        # 짝 추정식: 문턱 위 swap 판독마다 그 짝 target 이 문턱 아래(또는 안 읽힘)인가, 둘 다 위인데 swap 이 더 높은가
        e_only = sum(1 for e in S["E_sw"] if sc.get(inv[e], -1) < cut_p)
        e_win = sum(1 for e in S["E_sw"] if sc.get(inv[e], -1) >= cut_p and sc[e] > sc[inv[e]])
        res[n] = dict(arabidopsis=estimators(n_t, len(S["E_sp"]), r_sp),
                      swap=estimators(n_t, len(S["E_sw"]), r_sw, e_only, e_win),
                      swap_examples=[f"{inv[e]}->{e} {sc[e]:.3f}" for e in sorted(S["E_sw"], key=lambda x: -sc[x])[:8]])
        print(n, json.dumps(res[n], ensure_ascii=False), flush=True)
    os.makedirs("out/quant/validation", exist_ok=True)
    json.dump(res, open(f"out/quant/validation/entrapment_formal_{a.study}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("->", f"out/quant/validation/entrapment_formal_{a.study}.json")


if __name__ == "__main__":
    main()
