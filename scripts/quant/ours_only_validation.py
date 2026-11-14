# -*- coding: utf-8 -*-
"""ours_only_validation.py — '우리만' 인산화 펩타이드를 모델 바깥의 근거로 검증한다 (2026-11-02, 사용자 요청).

집단 (인산화 그룹 FDR 1% 통과, N말단 미표지 판독 제외, 펩타이드마다 최고 점수 인산화 PSM 하나):
  공유      = target 이고 Sequest·CHIMERYS·FragPipe 중 하나라도 찾은 서열 (정답 기준선)
  우리만-확신 = 공동 문턱 이상 · 길이 ≥ 7
  우리만-6   = 공동 문턱 이상 · 길이 6
  우리만-구제 = 공동 문턱 미만 (인산화 그룹 FDR 로만 통과)
  오답 기준선 = decoy 서열 · entrapment(애기장대) 서열에 맞은 판독 — 판독이 틀렸다는 것을 아는 집단

검증 (모두 모델이 쓰지 않은 정보):
  A entrapment  애기장대 Swiss-Prot 소화물(생쥐 target·decoy 와 겹치는 것 제외)에 맞은 판독 수로 집단별 오류율을 추정
  B 정량 일치    TMT 채널 패턴이 같은 단백질의 DB 확인 펩타이드 패턴과 얼마나 닮았나 (무작위 단백질 대비)
  C RT 일치      DB 확인 펩타이드로 분획별 RT 모델을 학습, 예측 오차를 집단별로 비교
  D 자리 타당성  UniProt 에 알려진 인산화 자리 비율, 키나아제 모티프(P+1, R−3, 산성) 비율 — 같은 단백질의 무작위 S/T/Y 대비
  E 재현성       같은 서열을 지지하는 PSM 수, 분획 수, 자리 일치율
B·C·D 는 정답 비율 추정치를 함께 낸다: π = (집단 − 오답 기준) / (공유 − 오답 기준).
F 합성 펩타이드 후보 패널(층화 추출)도 쓴다.

    python scripts/quant/ours_only_validation.py --study study3
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import count_matched_fdr as cm                # noqa: E402
import chimeric_case_study as cs              # noqa: E402
import compare_phospho_split as cps           # noqa: E402
import fragpipe_phospho_compare as fpc        # noqa: E402
from rescued_pick import FA, P, BASES         # noqa: E402

ENT = "out/fasta/entrapment/arath_UP000006548.fasta"
MODRES = "out/fasta/entrapment/mouse_modres.tsv.gz"
FP_ST = {"study3": "s3", "study2": "s2", "study1": "s1"}
GROUPS = ["shared", "ours_conf", "ours_resc", "wrong"]   # 2026-11-13: ours_resc 는 비워 둔다 — 인산화 그룹 기준 하나로 통일(사용자 지시)
rng = random.Random(0)


# ------------------------------------------------------------------ 읽기

def engine_phospho(study):
    """2026-11-13: DB 엔진이 찾은 인산화 bare 서열 — Fig. 4C 와 같은 기준.
    SEQUEST HT·CHIMERYS = PD PeptideGroups 인산화, FragPipe = 자기 기본 출력(psm.tsv) 인산화."""
    other = set()
    for suf in ("_PeptideGroups.txt", "-(1)_PeptideGroups.txt"):
        other |= cps.read_pd(P + BASES[study] + suf)[0]
    other |= fpc.phospho_bare(fpc.read_default(FP_ST[study]))[0]
    return other

def read_psms(study):
    """N말단 표지(또는 아세틸) 판독만. dict(stem, scan, pf, bare, score, z, ch(np.array) )"""
    out = []
    for p in sorted(glob.glob(f"out/quant/{study}/*_psm.tsv")):
        rd = csv.reader((ln for ln in open(p, encoding="utf-8") if not ln.startswith("#")), delimiter="\t")
        hdr = next(rd)
        i0 = hdr.index("in_db") + 1
        for c in rd:
            try:
                sc = float(c[4])
            except (ValueError, IndexError):
                continue
            pf = c[2]
            if not pf.startswith("["):
                continue
            try:
                pur = float(c[7])
            except ValueError:
                pur = float("nan")
            out.append(dict(stem=c[0], scan=int(c[1]), pf=pf, bare=cps.bare(pf), score=sc, z=int(c[6]), purity=pur,
                            ch=np.array([float(x) if x not in ("", "nan") else np.nan for x in c[i0:]], float)))
    return out


def read_rt(study):
    rt = {}
    for p in glob.glob(f"out/quant/{study}/*.rt.tsv"):
        st = os.path.basename(p)[:-len(".rt.tsv")]
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
            rt[(st, int(r["scan"]))] = float(r["rt_min"])
    return rt


def proteins(path):
    return {cm.accession_of(h): s for h, s in cm.read_fasta([path]) if s}


def full_digest(seqs, rev=False):
    out = set()
    for s in seqs:
        for pep in cm.digest(s[::-1] if rev else s, 2, 6, 50, met_loss=True):
            out.add(cm.normalize(pep))
    return out


def sites(pf):
    seq, mods = cs.our_form(pf)
    return seq, sorted(k for k, v in mods.items() if v == "Phospho" and k > 0)


# ------------------------------------------------------------------ 분석
def entrapment(best_p, groups_of, cut_p, joint, ent_hits, r_ratio, n_by_group, dec_hits):
    """집단 정의(문턱 대역·길이)를 오답 판독에도 그대로 적용해 센다."""
    def band(k, s):
        return "ours_conf"
    e = collections.Counter(); d = collections.Counter()
    for k in ent_hits:
        s = best_p[k][0]
        if s >= cut_p:
            e[band(k, s)] += 1; e["all"] += 1
    for k in dec_hits:
        s = best_p[k][0]
        if s >= cut_p:
            d[band(k, s)] += 1; d["all"] += 1
    res = {}
    for g in ["ours_conf", "ours_resc", "all"]:
        n = n_by_group["ours_all" if g == "all" else g]
        est_e = e[g] / r_ratio
        res[g] = dict(n_ours_only=n, entrap_hits=e[g], est_false_from_entrap=round(est_e, 1),
                      fdp_entrap=round(est_e / max(n, 1), 4), decoy_hits=d[g], fdp_decoy=round(d[g] / max(n, 1), 4))
    return res


ACTIVE = None          # 시료가 실제로 든 채널 (Study 3 은 16 중 12) — main 에서 정한다
ITEM_R = {}            # 펩타이드별 정량 상관 (G 에서 학습 펩타이드 여부와 엮는다)
LOAD = 0.0             # 채널 로딩 보정 = 확신 PSM 의 (행 중심화한) log2 값의 채널별 중앙값


def profile(ch):
    if ch is None:
        return None
    c = ch[ACTIVE]
    if not np.all(np.isfinite(c)) or np.any(c <= 0):
        return None
    v = np.log2(c)
    v = v - v.mean() - LOAD
    return v - v.mean()


def quant_concordance(rows_by_group, ref_rows, prot_of, n_null=1):
    """rows: (bare, profile, log10 리포터 세기 합). ref_rows: {bare: profile} — DB 확인 펩타이드. prot_of: {bare: set(acc)}
    '우리만' 은 대부분 PSM 1개짜리 저풍부 펩타이드라 리포터가 약해 상관이 낮게 나온다 → 공유 펩타이드의 (세기 5분위 × 분리 순도 4분위 × 기준 펩타이드 수 3구간) 칸별
    평균 상관으로 '같은 세기의 정답이면 기대되는 상관'을 만들고, π = (관측 − 무작위) / (세기 맞춘 기대 − 무작위)."""
    by_prot = collections.defaultdict(dict)
    for k, v in ref_rows.items():
        for a in prot_of.get(k, ()):
            by_prot[a][k] = v
    prots = [a for a, d in by_prot.items() if len(d) >= 3]

    def ref_for(accs, excl, with_n=False):
        vs = [v for a in accs for kk, v in by_prot.get(a, {}).items() if kk != excl]
        r = np.median(np.array(vs), axis=0) if len(vs) >= 3 else None
        return (r, len(vs)) if with_n else r

    per = {}
    null_all = []
    for g, items in rows_by_group.items():
        rs, it, nul, pu, nr = [], [], [], [], []
        for k, v, li, pur in items:
            ref, nref = ref_for(prot_of.get(k, ()), k, with_n=True)
            if ref is None:
                continue
            rs.append(float(np.corrcoef(v, ref)[0, 1])); it.append(li); pu.append(pur); nr.append(nref)
            ITEM_R[k] = rs[-1]
            for _ in range(n_null):
                rr = None
                while rr is None:          # 2026-11-13: 펩타이드 3개 미만 단백질이 뽑히면 다시 뽑는다
                    rr = ref_for([rng.choice(prots)], k)
                nul.append(float(np.corrcoef(v, rr)[0, 1]))
        null_all += nul
        per[g] = (np.array(rs), np.array(it), np.array(nul), np.array(pu), np.array(nr))
    m_n = float(np.mean(null_all))
    rs_s, it_s, _, pu_s, nr_s = per["shared"]
    bn = lambda x: np.searchsorted([6, 16], x, side="right")          # 기준 펩타이드 수 3–5 / 6–15 / 16+
    ei = np.quantile(it_s, np.linspace(0, 1, 6)); ei[0], ei[-1] = -np.inf, np.inf
    ep = np.quantile(np.nan_to_num(pu_s), np.linspace(0, 1, 5)); ep[0], ep[-1] = -np.inf, np.inf
    bi = lambda x: np.clip(np.searchsorted(ei, x, side="right") - 1, 0, 4)
    bp = lambda x: np.clip(np.searchsorted(ep, np.nan_to_num(x), side="right") - 1, 0, 3)
    cell = collections.defaultdict(list)
    for r, i_, p_, n_ in zip(rs_s, bi(it_s), bp(pu_s), bn(nr_s)):
        cell[(i_, p_, n_)].append(r)
    cmean = {c: float(np.mean(v)) for c, v in cell.items()}
    out = {}
    for g, (rs, it, nul, pu, nr) in per.items():
        if not len(rs):
            out[g] = dict(n=0); continue
        exp = float(np.mean([cmean.get((i_, p_, n_), rs_s.mean()) for i_, p_, n_ in zip(bi(it), bp(pu), bn(nr))]))
        out[g] = dict(n=len(rs), mean_r=round(float(rs.mean()), 3), median_log10_intensity=round(float(np.median(it)), 2),
                      median_purity=round(float(np.nanmedian(pu)), 3), median_n_ref=float(np.median(nr)),
                      expected_r_if_correct=round(exp, 3), null_mean_r=round(float(nul.mean()), 3))
        if g != "shared":
            out[g]["pi_correct_raw"] = round((rs.mean() - m_n) / (rs_s.mean() - m_n), 3)
            out[g]["pi_correct_matched"] = round((rs.mean() - m_n) / (exp - m_n), 3)
    out["_null_mean_r_pooled"] = round(m_n, 3)
    return out


AAS = "ACDEFGHKLMNPQRSTVWY"


def rt_features(pf):
    seq, mods = cs.our_form(pf)
    f = [seq.count(a) for a in AAS]
    f += [sum(1 for k, v in mods.items() if v == "Phospho" and seq[k - 1] == x) for x in "STY"]
    f += [sum(1 for v in mods.values() if v == "Oxidation"), len(seq), int(mods.get(0) == "Acetyl")]
    f += [int(seq[0] == a) for a in AAS] + [int(seq[-1] == a) for a in "KR"]
    f += [seq[:3].count(a) for a in "DEKRH"] + [seq[-3:].count(a) for a in "DEH"]
    return f


def rt_concordance(train_by_stem, test_by_stem):
    """분획마다 HistGradientBoosting. 공유는 5겹 교차 예측(out-of-fold). 지표 = |오차| ≤ 공유 오차 95분위 비율."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    resid = collections.defaultdict(list)
    shuffled = []
    for st, tr in train_by_stem.items():
        if len(tr) < 200:
            continue
        X = np.array([rt_features(pf) for pf, _ in tr], float); y = np.array([t for _, t in tr])
        oof = np.zeros(len(y))
        for a, b in KFold(5, shuffle=True, random_state=0).split(X):
            m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X[a], y[a])
            oof[b] = m.predict(X[b])
        resid["shared"] += list(np.abs(oof - y))
        shuffled += list(np.abs(oof - rng.sample(list(y), len(y))))        # 무작위 RT 짝짓기 = 완전 무관 기준
        m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X, y)
        for g, items in test_by_stem.get(st, {}).items():
            if items:
                p = m.predict(np.array([rt_features(pf) for pf, _ in items], float))
                resid[g] += list(np.abs(p - np.array([t for _, t in items])))
    q95 = float(np.quantile(resid["shared"], 0.95))
    out = {}
    within = lambda v: float(np.mean(np.array(v) <= q95)) if v else None
    f_rand = within(shuffled)
    for g, v in resid.items():
        w = within(v)
        out[g] = dict(n=len(v), median_abs_err_min=round(float(np.median(v)), 2), within_q95=round(w, 3))
    out["_random_pairing"] = dict(n=len(shuffled), median_abs_err_min=round(float(np.median(shuffled)), 2), within_q95=round(f_rand, 3))
    out["_q95_min"] = round(q95, 2)
    # 2026-11-13: Fig. 5C 누적분포용 — 오차를 공유 95분위로 나눈 값(1 = 경계)
    out["_rel_err"] = {g: [round(x / q95, 4) for x in v] for g, v in resid.items()}
    for g in out:
        if not g.startswith("_") and g != "shared":
            out[g]["pi_correct_vs_random"] = round((out[g]["within_q95"] - f_rand) / (out["shared"]["within_q95"] - f_rand), 3)
            if "wrong" in out and out["wrong"]["n"] >= 20:
                fw = out["wrong"]["within_q95"]
                out[g]["pi_correct_vs_wrong"] = round((out[g]["within_q95"] - fw) / (out["shared"]["within_q95"] - fw), 3)
    return out


def site_plausibility(items_by_group, prot_seq, prot_of, known):
    """items: (bare, best pf). 단백질 위치 = 정규화 서열에서 첫 일치.
    기준선은 집단마다 따로: 같은 단백질에서 '펩타이드 안 위치가 같은' 무작위 S/T/Y
    (그 자리에서 site−1 칸 앞이 K/R 또는 단백질 시작) — 짧은 펩타이드일수록 절단 자리 R/K 가 −2/−3 에 오는 위치 편향을 맞춘다.
    정답 비율 π = (관측 − 기준) / (공유 관측 − 공유 기준)."""
    def ctx(s, i):
        return "".join(s[j] if 0 <= j < len(s) else "_" for j in range(i - 7, i + 8))

    def motif(c):
        return dict(P1=c[8] == "P", R3=c[4] == "R", acid=sum(1 for x in c[9:12] if x in "DE") >= 2)
    keys = ("known", "P1", "R3", "acid")
    out = {}
    for g, items in items_by_group.items():
        cnt = collections.Counter(); nul = collections.Counter(); n = nn = 0
        for k, pf in items:
            seq, st = sites(pf)
            hit = None
            for a in sorted(prot_of.get(k, ())):
                s = prot_seq.get(a)
                if not s:
                    continue
                pos = cm.normalize(s).find(k)
                if pos >= 0:
                    hit = (a, s, pos); break
            if not hit or not st:
                continue
            a, s, pos = hit
            for site in st:
                i = pos + site - 1
                n += 1
                cnt["known"] += (a, i + 1) in known
                for mk, mv in motif(ctx(s, i)).items():
                    cnt[mk] += mv
                cands = [j for j, x in enumerate(s) if x in "STY" and j != i and (j - site < 0 or s[j - site] in "KR")
                         and j - site >= -1]
                if cands:
                    j = rng.choice(cands); nn += 1
                    nul["known"] += (a, j + 1) in known
                    for mk, mv in motif(ctx(s, j)).items():
                        nul[mk] += mv
        out[g] = dict(n_sites=n, **{m: round(cnt[m] / max(n, 1), 3) for m in keys},
                      **{"null_" + m: round(nul[m] / max(nn, 1), 3) for m in keys})
    sh = out["shared"]
    for g in out:
        if g != "shared":
            for m in keys:
                d = sh[m] - sh["null_" + m]
                out[g]["pi_" + m] = round((out[g][m] - out[g]["null_" + m]) / d, 3) if abs(d) > 0.02 else None
    return out


def reproducibility(psms_by_bare, groups, cut_p):
    out = {}
    for g, keys in groups.items():
        npsm, nfr, agree = [], [], []
        for k in keys:
            ps = [p for p in psms_by_bare.get(k, []) if p["score"] >= cut_p and "Phospho" in p["pf"]]
            if not ps:
                continue
            b = max(ps, key=lambda p: p["score"])
            st_b = sites(b["pf"])[1]
            npsm.append(len(ps)); nfr.append(len({p["stem"] for p in ps}))
            agree.append(np.mean([sites(p["pf"])[1] == st_b for p in ps]))
        out[g] = dict(n=len(npsm), median_psms=float(np.median(npsm)) if npsm else None,
                      frac_ge2_psms=round(float(np.mean(np.array(npsm) >= 2)), 3) if npsm else None,
                      median_fractions=float(np.median(nfr)) if nfr else None,
                      site_agreement=round(float(np.mean(agree)), 3) if agree else None)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="study3")
    a = ap.parse_args()
    psms = read_psms(a.study)
    print("PSM", len(psms), flush=True)
    best, best_p, by_bare = {}, {}, collections.defaultdict(list)
    for p in psms:
        k = p["bare"]
        by_bare[k].append(p)
        if p["score"] > best.get(k, (-1,))[0]:
            best[k] = (p["score"], p)
        if "Phospho" in p["pf"] and p["score"] > best_p.get(k, (-1,))[0]:
            best_p[k] = (p["score"], p)
    tgt, dec = cm.target_decoy(FA, set(best))
    joint, _ = cm.cutoff_for_fdr({k: v[0] for k, v in best.items()}, tgt, dec, 0.01)
    cut_p, _ = cm.cutoff_for_fdr({k: v[0] for k, v in best_p.items()}, tgt, dec, 0.01)
    print(f"joint {joint:.4f}  phospho {cut_p:.4f}", flush=True)

    # entrapment 소화물 (생쥐 target·decoy 와 겹치는 펩타이드 제외)
    mouse = proteins(FA)
    t_all = full_digest(mouse.values()); d_all = full_digest(mouse.values(), rev=True)
    ent = proteins(ENT)
    e_all = full_digest(ent.values()) - t_all - d_all
    r_ratio = len(e_all) / len(t_all)
    print(f"digest target {len(t_all):,}  entrap-only {len(e_all):,}  r={r_ratio:.3f}", flush=True)
    ent_hits = {k for k in best_p if k in e_all and k not in tgt and k not in dec}
    dec_hits = {k for k in best_p if k in dec}
    del t_all, d_all

    other = engine_phospho(a.study)   # 2026-11-13: Fig. 4C 기준 (엔진의 인산화 동정)

    groups = {g: [] for g in GROUPS}
    for k, (s, p) in best_p.items():
        if s < cut_p:
            continue
        if k in tgt:
            if k in other:
                groups["shared"].append(k)
            else:
                groups["ours_conf"].append(k)
    # 오답 기준선: decoy·entrapment 판독. 문턱 위만으로는 수가 적어 점수 전 범위를 쓴다(판독이 틀렸다는 사실은 같다)
    wrong_all = sorted(ent_hits | dec_hits)
    groups["wrong"] = wrong_all
    n_by = {g: len(v) for g, v in groups.items()}
    n_by["ours_all"] = n_by["ours_conf"] + n_by["ours_resc"]
    print("groups", n_by, flush=True)
    res = dict(study=a.study, joint_cut=joint, phospho_cut=cut_p, groups=n_by, entrap_ratio=round(r_ratio, 4))

    # A
    res["A_entrapment"] = entrapment(best_p, groups, cut_p, joint, ent_hits, r_ratio, n_by, dec_hits)
    print("A", json.dumps(res["A_entrapment"], ensure_ascii=False), flush=True)

    # B 정량: 활성 채널 = 확신 PSM 의 절반 이상에서 신호가 있는 채널
    global ACTIVE
    chs = np.array([p["ch"] for p in psms if p["score"] >= joint])
    ACTIVE = np.where(np.mean(np.nan_to_num(chs) > 0, axis=0) >= 0.5)[0]
    global LOAD
    full = chs[:, ACTIVE]
    full = full[np.all(np.isfinite(full) & (full > 0), axis=1)]
    lv = np.log2(full); lv -= lv.mean(axis=1, keepdims=True)
    LOAD = np.median(lv, axis=0)
    res["active_channels"] = ACTIVE.tolist()
    print("active channels", ACTIVE.tolist(), flush=True)
    # 기준 = DB 확인 펩타이드(인산화·비인산화, 공동 문턱 이상)의 최고 점수 PSM
    prot_of = {k: set(v) for k, v in tgt.items()}
    ref = {}
    for k, (s, p) in best.items():
        if k in tgt and k in other and s >= joint:
            v = profile(p["ch"])
            if v is not None:
                ref[k] = v
    qb = {}
    for g in ["shared", "ours_conf", "ours_resc"]:
        qb[g] = [(k, v, float(np.log10(np.nansum(best_p[k][1]["ch"][ACTIVE]))), best_p[k][1]["purity"])
                 for k in groups[g] for v in [profile(best_p[k][1]["ch"])] if v is not None]
    res["B_quant"] = quant_concordance(qb, ref, prot_of)
    print("B", json.dumps(res["B_quant"], ensure_ascii=False), flush=True)

    # C RT
    rt = read_rt(a.study)
    if rt:
        train = collections.defaultdict(list); test = collections.defaultdict(lambda: collections.defaultdict(list))
        for g in GROUPS:
            for k in groups[g]:
                p = best_p[k][1]
                t = rt.get((p["stem"], p["scan"]))
                if t is None:
                    continue
                (train[p["stem"]].append((p["pf"], t)) if g == "shared" else test[p["stem"]][g].append((p["pf"], t)))
        res["C_rt"] = rt_concordance(train, test)
        print("C", json.dumps(res["C_rt"], ensure_ascii=False), flush=True)
    else:
        res["C_rt"] = "RT 표 없음 (extract_rt.py 먼저)"

    # D 자리
    known = set()
    with gzip.open(MODRES, "rt", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            acc, _, ft = line.rstrip("\n").partition("\t")
            for pos, note in re.findall(r"MOD_RES (\d+); /note=\"(Phospho[a-z]+)", ft):
                known.add((acc, int(pos)))
    items = {g: [(k, best_p[k][1]["pf"]) for k in groups[g]] for g in ["shared", "ours_conf", "ours_resc"]}
    res["D_site"] = site_plausibility(items, mouse, prot_of, known)
    print("D", json.dumps(res["D_site"], ensure_ascii=False), flush=True)

    # I 자리 신규성별 정량 상관: 새 자리(UniProt 미등재)는 단백질 양과 따로 움직일 수 있다 — 공유 펩타이드에서도 그런지 본다
    def all_known(k):
        seq, st = sites(best_p[k][1]["pf"])
        for acc in sorted(prot_of.get(k, ())):
            sq = mouse.get(acc)
            pos = cm.normalize(sq).find(k) if sq else -1
            if pos >= 0:
                return all((acc, pos + x) in known for x in st)
        return False
    ires = {}
    for g in ["shared", "ours_conf", "ours_resc"]:
        kn = [ITEM_R[k] for k in groups[g] if k in ITEM_R and all_known(k)]
        nv = [ITEM_R[k] for k in groups[g] if k in ITEM_R and not all_known(k)]
        ires[g] = dict(r_known=round(float(np.mean(kn)), 3) if kn else None, n_known=len(kn),
                       r_novel=round(float(np.mean(nv)), 3) if nv else None, n_novel=len(nv))
    res["I_quant_by_site_novelty"] = ires
    print("I", json.dumps(ires, ensure_ascii=False), flush=True)

    # J·K: "인산화 자리는 원래 단백질과 따로 움직인다" 반론 점검 (2026-11-03 사용자 지적)
    def prot_sites(k, pf):
        """(대표 단백질, {단백질 내 인산화 위치}, 모티프 분류)"""
        seq, st = sites(pf)
        for acc in sorted(prot_of.get(k, ())):
            sq = mouse.get(acc)
            pos = cm.normalize(sq).find(k) if sq else -1
            if pos >= 0 and st:
                ps = frozenset(pos + x for x in st)
                i = pos + st[0] - 1
                cls = "P1" if i + 1 < len(sq) and sq[i + 1] == "P" else ("R3" if i - 3 >= 0 and sq[i - 3] == "R" else "other")
                return acc, ps, cls
        return None
    # J 모티프별 상관: 공유에서도 R−3(염기성 키나아제) 자리는 단백질 기준과 상관이 낮은가?
    jres = {}
    for g in ["shared", "ours_conf", "ours_resc"]:
        byc = collections.defaultdict(list)
        for k in groups[g]:
            if k in ITEM_R:
                ps = prot_sites(k, best_p[k][1]["pf"])
                if ps:
                    byc[ps[2]].append(ITEM_R[k])
        jres[g] = {c: dict(n=len(v), mean_r=round(float(np.mean(v)), 3)) for c, v in sorted(byc.items())}
    sh = jres["shared"]
    for g in ["ours_conf", "ours_resc"]:          # 우리만의 모티프 구성으로 공유 상관을 다시 가중한 기대값
        n_tot = sum(d["n"] for d in jres[g].values())
        if n_tot:
            exp = sum(d["n"] * sh.get(c, {}).get("mean_r", 0.0) for c, d in jres[g].items()) / n_tot
            obs = sum(d["n"] * d["mean_r"] for d in jres[g].values()) / n_tot
            jres[g]["_motif_weighted_expected_r"] = round(exp, 3)
            jres[g]["_pi_motif_matched"] = round((obs - res["B_quant"]["_null_mean_r_pooled"]) /
                                                 (exp - res["B_quant"]["_null_mean_r_pooled"]), 3)
    res["J_quant_by_motif"] = jres
    print("J", json.dumps(jres, ensure_ascii=False), flush=True)

    # K 같은 자리 짝: 우리만 펩타이드와 같은 인산화 자리를 담은 다른 DB 확인 인산화 펩타이드(미절단·산화 변형 등).
    #   같은 인산화 분자종이라 생물학적으로 패턴이 달라질 이유가 없다 → 맞게 읽었으면 상관이 높아야 한다.
    site_ref = collections.defaultdict(list)
    for kk, v in ref.items():
        pf = best[kk][1]["pf"]
        if "Phospho" not in pf:
            continue
        ps = prot_sites(kk, pf)
        if ps:
            site_ref[(ps[0], ps[1])].append((kk, v))
    all_keys = list(site_ref)
    kres = {}
    for g in ["shared", "ours_conf", "ours_resc"]:
        rs, nul, rprot = [], [], []
        for k in groups[g]:
            pf = best_p[k][1]["pf"]
            v = profile(best_p[k][1]["ch"])
            ps = prot_sites(k, pf)
            if v is None or not ps:
                continue
            mates = [mv for kk, mv in site_ref.get((ps[0], ps[1]), []) if kk != k]
            if not mates:
                continue
            rs.append(float(np.corrcoef(v, np.median(np.array(mates), axis=0))[0, 1]))
            if k in ITEM_R:
                rprot.append(ITEM_R[k])              # 같은 부분집합의 단백질 기준 상관 (부분집합 선택 편향 점검)
            rk = rng.choice(all_keys)
            nul.append(float(np.corrcoef(v, np.median(np.array([mv for _, mv in site_ref[rk]]), axis=0))[0, 1]))
        kres[g] = dict(n=len(rs), mean_r=round(float(np.mean(rs)), 3) if rs else None,
                       median_r=round(float(np.median(rs)), 3) if rs else None,
                       null_mean_r=round(float(np.mean(nul)), 3) if nul else None,
                       protein_ref_r_same_subset=round(float(np.mean(rprot)), 3) if rprot else None, n_prot=len(rprot))
    s0 = kres["shared"]
    for g in ["ours_conf", "ours_resc"]:
        if kres[g]["n"] and s0["n"]:
            kres[g]["pi_correct"] = round((kres[g]["mean_r"] - kres[g]["null_mean_r"]) / (s0["mean_r"] - s0["null_mean_r"]), 3)
    res["K_same_site_partner"] = kres
    print("K", json.dumps(kres, ensure_ascii=False), flush=True)

    # E 재현성
    res["E_repro"] = reproducibility(by_bare, groups, cut_p)
    print("E", json.dumps(res["E_repro"], ensure_ascii=False), flush=True)

    # G 학습 펩타이드 여부: 모델이 학습 때 본 실제 펩타이드로 '잘못 읽으면' 늘 target 에 맞아 decoy·entrapment 로 안 잡힌다.
    #   그런 오류라면 학습 펩타이드인 '우리만' 의 정량 상관이 낮아야 한다.
    import pickle
    train = {cm.normalize(x) for x in pickle.load(open("out/train_peptides_true.pkl", "rb"))}
    gres = {}
    for g in ["shared", "ours_conf", "ours_resc", "wrong"]:
        ks = groups[g]
        it = [k for k in ks if k in train]
        r_in = [ITEM_R[k] for k in it if k in ITEM_R]
        r_out = [ITEM_R[k] for k in ks if k not in train and k in ITEM_R]
        gres[g] = dict(n=len(ks), frac_in_train=round(len(it) / max(len(ks), 1), 3),
                       quant_mean_r_in_train=round(float(np.mean(r_in)), 3) if r_in else None, n_r_in=len(r_in),
                       quant_mean_r_not_in_train=round(float(np.mean(r_out)), 3) if r_out else None, n_r_out=len(r_out))
    res["G_train_membership"] = gres
    print("G", json.dumps(gres, ensure_ascii=False), flush=True)

    # H 같은 스캔의 다른 동정: 우리 판독 스캔에서 Sequest·CHIMERYS·FragPipe 가 '다른' 펩타이드를 잡았나 (키메라의 약한 쪽을 읽은 경우)
    #   그런 스캔이면 리포터는 강한 쪽 펩타이드 몫이라 우리 단백질과 정량 상관이 낮아도 판독 오류는 아니다.
    scan_other = collections.defaultdict(set)
    for suf in ("_", "-(1)_"):
        base = P + BASES[a.study] + suf
        fmap = {}
        for r in csv.DictReader(open(base + "InputFiles.txt", encoding="utf-8-sig", errors="replace"), delimiter="	"):
            fn = r["File Name"].replace("\\", "/").split("/")[-1]
            if fn.lower().endswith(".raw"):
                fmap[r["Study File ID"]] = fn[:-4]
        for r in csv.DictReader(open(base + "PSMs.txt", encoding="utf-8-sig", errors="replace"), delimiter="	"):
            st = fmap.get(r["File ID"])
            if st:
                pk = cm.normalize(re.sub(r"^\[.*?\]\.|\.\[.*?\]$", "", r["Annotated Sequence"]).upper())
                scan_other[(st, int(r["First Scan"]))].add(pk)
    for p in fpc.read_default(FP_ST[a.study]):   # 2026-11-13: FragPipe 기본 출력
        scan_other[(p["stem"], p["scan"])].add(p["bare"])
    hres = {}
    for g in ["shared", "ours_conf", "ours_resc"]:
        flag, r_c, r_n = 0, [], []
        for k in groups[g]:
            bp_ = best_p[k][1]
            oth = scan_other.get((bp_["stem"], bp_["scan"]), set()) - {k}
            if oth:
                flag += 1
            if k in ITEM_R:
                (r_c if oth else r_n).append(ITEM_R[k])
        hres[g] = dict(n=len(groups[g]), frac_scan_has_other_id=round(flag / max(len(groups[g]), 1), 3),
                       quant_r_when_other=round(float(np.mean(r_c)), 3) if r_c else None, n_other=len(r_c),
                       quant_r_when_alone=round(float(np.mean(r_n)), 3) if r_n else None, n_alone=len(r_n))
    res["H_same_scan_other_id"] = hres
    print("H", json.dumps(hres, ensure_ascii=False), flush=True)

    # F 합성 후보 패널: 우리만-확신 길이층별 + 6-mer + 구제 + 공유 대조
    panel = []
    def take(keys, n, label):
        ks = sorted(keys, key=lambda k: -best_p[k][0])
        step = max(1, len(ks) // n)
        for k in ks[::step][:n]:
            s, p = best_p[k]
            seq, st = sites(p["pf"])
            panel.append(dict(group=label, peptidoform=p["pf"], score=round(s, 4), length=len(k), stem=p["stem"], scan=p["scan"],
                              proteins=";".join(sorted(prot_of.get(k, ()))), sites=",".join(f"{seq[i - 1]}{i}" for i in st),
                              quant_r=round(ITEM_R[k], 3) if k in ITEM_R else "", in_train=int(k in train)))
    conf = groups["ours_conf"]
    take([k for k in conf if len(k) <= 8], 8, "ours_conf_len7-8")
    take([k for k in conf if 9 <= len(k) <= 11], 8, "ours_conf_len9-11")
    take([k for k in conf if len(k) >= 12], 4, "ours_conf_len12+")
    take(groups["ours_resc"], 5, "ours_resc")
    take(groups["shared"], 5, "shared_control")
    os.makedirs("out/quant/validation", exist_ok=True)
    with open(f"out/quant/validation/synthetic_panel_{a.study}.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(panel[0]), delimiter="\t"); w.writeheader(); w.writerows(panel)
    json.dump(res, open(f"out/quant/validation/ours_only_validation_{a.study}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("-> out/quant/validation/", flush=True)


if __name__ == "__main__":
    main()
