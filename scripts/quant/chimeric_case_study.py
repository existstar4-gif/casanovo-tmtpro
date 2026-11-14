# -*- coding: utf-8 -*-
"""chimeric_case_study.py — Study 2 phospho (학습에 쓰지 않은 스터디) 에서 두 가지를 잰다 (2026-10-21).

(A) 키메라 정량 분리 가능성
    CHIMERYS 가 한 스캔에서 펩타이드 둘 이상을 동정해도 PD 는 그 스캔의 리포터 18개 값을 모두에게 복사한다
    (an in-house phospho study, fr_1: 키메라 8,204 스캔 전부 동일 벡터). MS2 리포터만으로는 나눌 수 없다.
    대안: 짝 펩타이드 A 가 다른 스캔(12개 분획 어디든 — 같은 18-plex 시료)에서 깨끗하게 측정됐다면 A 의
    채널 비율 pA 를 알 수 있다. 키메라 스캔 관측 O 에서 A 의 몫(MS1 분리창 안 A 의 세기 비율 sA)을 빼
    B 를 추정한다:  B_hat = O - sA * sum(O) * pA.
    ① 가능성: 키메라 스캔 중 짝에게 깨끗한 기준이 있는 비율
    ② 검증: 펩타이드 둘이고 둘 다 기준이 있는 스캔에서, 추정 B_hat 이 B 의 깨끗한 기준 pB 에
       '그대로 쓴 O' 보다 가까워지는가 (log2 채널비 오차, 상관).

(B) CHIMERYS 가 PSM 을 낸 스캔을 우리 모델이 왜 못 맞혔나 — fr_01, GUI 실행 결과(psms.tsv) 기준.
    분류: 일치 / 인산화 자리만 다름 / 다른 수식 / 우리 입력에서 빠짐 / CHIMERYS 가 목표 전구체가 아닌
    같이 분리된 전구체만 동정 / 우리 판독이 전구체 질량 불일치 / 질량은 맞는 다른 서열.
    각 군의 길이·전하·인산화 수·TMT 미표지·분리창 순도·조각 이온 커버리지를 비교한다.

    python scripts/quant/chimeric_case_study.py --out out/quant/chimeric_case_study
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from pyteomics import mzml

sys.path.insert(0, os.environ.get("CASANOVO_STUDIO", "casanovo_studio"))
from core import reporters as rp          # noqa: E402  (purity / 리포터 m/z 를 정량 모듈과 같게)

csv.field_size_limit(10 ** 9)
PD_DIR = Path(r"data/pd_heldout")
PD_CHIM = PD_DIR / "study2_phospho_fr_-(1)_PSMs.txt"
PD_SEQ = PD_DIR / "study2_phospho_fr__PSMs.txt"
MZDIR = Path(r"data/mzml_tmt")
STEM = "study2_phospho_fr_{:02d}"
OURS = Path(r"data/demo/study2_phospho_fr01\psms.tsv")

PROTON, H2O, C13 = 1.007276, 18.010565, 1.0033548378
H3PO4 = 97.976896
AA = {"G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841, "T": 101.04768, "C": 103.00919,
      "L": 113.08406, "I": 113.08406, "N": 114.04293, "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259,
      "M": 131.04049, "H": 137.05891, "F": 147.06841, "R": 156.10111, "Y": 163.06333, "W": 186.07931}
MODM = {"TMTpro": 304.207146, "Phospho": 79.966331, "Oxidation": 15.994915, "Carbamidomethyl": 57.021464,
        "Acetyl": 42.010565, "Deamidated": 0.984016, "Carbamyl": 43.005814, "Ammonia-loss": -17.026549}
CH18 = rp.channels(18)
FRAG_TOL = 0.02


# ------------------------------------------------------------------ peptidoform keys
def pd_form(annot, mods):
    """PD -> (seq, {pos: mod}) ; pos 0 = N-term, 1..n residues. Carbamidomethyl (고정) 은 뺀다."""
    seq = annot.split(".")[1] if annot.count(".") >= 2 else annot
    seq = seq.upper()
    out = {}
    for m in (mods or "").split(";"):
        m = m.strip()
        g = re.match(r"^(N-Term|C-Term|[A-Z]\d+)((?:\([^)]*\))+)$", m)
        if not g:
            continue
        name = re.findall(r"\(([^)]*)\)", g.group(2))[-1]
        if name == "Carbamidomethyl":
            continue
        pos = 0 if g.group(1) == "N-Term" else (len(seq) + 1 if g.group(1) == "C-Term" else int(g.group(1)[1:]))
        out[pos] = name
    return seq, out


def our_form(pep):
    nmods = {}
    m = re.match(r"^\[([^\]]+)\]-", pep)
    if m:
        nmods[0] = m.group(1)
        pep = pep[m.end():]
    seq = ""
    for k, (aa, mod) in enumerate(re.findall(r"([A-Z])(?:\[([^\]]+)\])?", pep), 1):
        seq += aa
        if mod and mod != "Carbamidomethyl":
            nmods[k] = mod
    return seq, nmods


def key(seq, mods):
    return seq.replace("I", "L") + "|" + ",".join(f"{p}{mods[p]}" for p in sorted(mods))


def mass_of(seq, mods):
    m = sum(AA[a] for a in seq) + H2O + sum(MODM.get(v, 0.0) for v in mods.values())
    m += 57.021464 * seq.count("C")
    return m


def frag_cover(seq, mods, z, mz, it):
    """절단 자리 커버리지(b/y, 1+..min(z-1,2)+, 인산 중성손실 포함)와 설명된 세기 비율(m/z >= 140)."""
    n = len(seq)
    res = [AA[a] + (57.021464 if a == "C" else 0.0) + MODM.get(mods.get(i + 1, ""), 0.0) for i, a in enumerate(seq)]
    res[0] += MODM.get(mods.get(0, ""), 0.0)
    ph = [1 if mods.get(i + 1) == "Phospho" else 0 for i in range(n)]
    sel = mz >= 140
    pm, pi = mz[sel], it[sel]
    tot = float(pi.sum()) or 1.0
    used = set()
    covered = 0
    for c in range(1, n):                       # cut after residue c
        bmass = sum(res[:c]); ymass = sum(res[c:]) + H2O
        nb, ny = sum(ph[:c]), sum(ph[c:])
        hit = False
        for base, nph in ((bmass, nb), (ymass, ny)):
            for q in range(1, max(1, min(z - 1, 2)) + 1):
                for loss in ([0.0, H3PO4] if nph else [0.0]):
                    t = (base - loss + q * PROTON) / q
                    j = int(np.searchsorted(pm, t - FRAG_TOL)); k2 = int(np.searchsorted(pm, t + FRAG_TOL))
                    if k2 > j:
                        hit = True
                        used.update(range(j, k2))
        covered += hit
    return covered / max(1, n - 1), float(pi[list(used)].sum()) / tot if used else 0.0


# ------------------------------------------------------------------ PD loading
def load_pd(path, engine):
    out = collections.defaultdict(list)
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            fr = int(r["File ID"].split(".")[-1])
            seq, mods = pd_form(r["Annotated Sequence"], r["Modifications"])
            z = int(r["Charge"])
            mh = float(r["Theo. MH+ [Da]"])
            out[(fr, int(r["First Scan"]))].append(dict(
                engine=engine, seq=seq, mods=mods, key=key(seq, mods), z=z, mono_mz=(mh - PROTON + z * PROTON) / z,
                mass=mh - PROTON, interf=float(r["Isolation Interference [%]"] or "nan"),
                prot=r.get("Master Protein Accessions", "")))
    return out


# ------------------------------------------------------------------ mzML pass
def scan_pass(fr, want_shares, keep_peaks):
    """한 분획 mzML -> {scan: dict(rep, nfound, sel_mz, z, purity, shares{key: s}, npk, [mz,it])}."""
    path = MZDIR / (STEM.format(fr) + ".mzML")
    info, ms1, order, last = {}, {}, [], None
    with mzml.read(str(path)) as rd:
        for sp in rd:
            lvl = sp.get("ms level")
            if lvl == 1:
                s1 = rp.scan_of(sp.get("id"))
                ms1[s1] = (np.asarray(sp["m/z array"]), np.asarray(sp["intensity array"]))
                order.append(s1); last = s1
                if len(order) > 100:
                    ms1.pop(order.pop(0), None)
                continue
            if lvl != 2:
                continue
            scan = rp.scan_of(sp.get("id"))
            mz = np.asarray(sp["m/z array"]); it = np.asarray(sp["intensity array"])
            prec = sp["precursorList"]["precursor"][0]
            sel = prec["selectedIonList"]["selectedIon"][0]
            smz = float(sel.get("selected ion m/z", 0.0)); z = int(sel.get("charge state", 0) or 0)
            win = prec.get("isolationWindow", {})
            center = float(win.get("isolation window target m/z", smz) or smz)
            lo = float(win.get("isolation window lower offset", 0.7) or 0.7)
            hi = float(win.get("isolation window upper offset", lo) or lo)
            ref = rp.scan_of(prec.get("spectrumRef")) or last
            vals = []
            for _n, t in CH18:
                j = int(np.searchsorted(mz, t - rp.DEFAULT_TOL)); k2 = int(np.searchsorted(mz, t + rp.DEFAULT_TOL))
                vals.append(float(it[j:k2].max()) if k2 > j else 0.0)
            d = dict(rep=vals, nfound=sum(v > 0 for v in vals), sel_mz=smz, z=z, npk=len(mz), center=center,
                     lo=lo, hi=hi)
            if ref in ms1:
                d["purity"] = rp.purity(*ms1[ref], smz, z, center, lo, hi)
                for k, (pmz, pz) in want_shares.get(scan, {}).items():
                    d.setdefault("shares", {})[k] = rp.purity(*ms1[ref], pmz, pz, center, lo, hi)
            else:
                d["purity"] = float("nan")
            if scan in keep_peaks:
                d["mz"], d["it"] = mz, it
            info[scan] = d
    return info


def l2r(v):
    v = np.asarray(v, float)
    v = np.where(v > 0, v, np.nan)
    return np.log2(v / np.nanmean(v))


def cmp_profiles(est, truth):
    a, b = l2r(est), l2r(truth)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10:
        return float("nan"), float("nan")
    return float(np.nanmean(np.abs(a[ok] - b[ok]))), float(np.corrcoef(a[ok], b[ok])[0, 1])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/quant/chimeric_case_study")
    ap.add_argument("--clean-purity", type=float, default=0.9)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    chim = load_pd(PD_CHIM, "CHIMERYS")
    seqh = load_pd(PD_SEQ, "Sequest")
    print("PD loaded", len(chim), len(seqh), flush=True)

    # 스캔별 동정 집합(두 엔진 합): 펩타이드가 하나뿐인 스캔만 '깨끗한 기준' 후보
    ids = collections.defaultdict(dict)
    for src in (chim, seqh):
        for k, lst in src.items():
            for p in lst:
                ids[k].setdefault(p["key"], p)
    chimeric = {k: v for k, v in chim.items() if len({p["key"] for p in v}) >= 2}

    want = collections.defaultdict(dict)        # (fr, scan) -> {key: (mono_mz, z)}
    for k, v in chimeric.items():
        for p in v:
            want[k][p["key"]] = (p["mono_mz"], p["z"])
    keep_fr1 = {s for (fr, s) in chim if fr == 1}

    scans = {}
    for fr in range(1, 13):
        w = {s: d for (f, s), d in want.items() if f == fr}
        scans[fr] = scan_pass(fr, w, keep_fr1 if fr == 1 else set())
        print(f"fr {fr:02d}: {len(scans[fr]):,} MS2", flush=True)

    # ---------------- (A) clean reference profiles
    refs = collections.defaultdict(list)
    for (fr, s), pk in ids.items():
        if len(pk) != 1:
            continue
        d = scans[fr].get(s)
        if not d or d["nfound"] < 18 or not (d["purity"] >= args.clean_purity):
            continue
        (kk,) = pk.keys()
        v = np.asarray(d["rep"]); refs[kk].append(v / v.sum())
    prof = {k: (np.median(np.vstack(v), axis=0) / np.median(np.vstack(v), axis=0).sum(), len(v)) for k, v in refs.items()}
    # 노이즈 바닥: 깨끗한 스캔 하나 vs 같은 펩타이드의 나머지 깨끗한 스캔 중앙값 (기준이 3개 이상인 펩타이드)
    noise = []
    for k, v in refs.items():
        if len(v) >= 3:
            for i in range(min(len(v), 5)):
                rest = np.median(np.vstack(v[:i] + v[i + 1:]), axis=0)
                noise.append(cmp_profiles(v[i], rest)[0])

    feas = collections.Counter()
    tests = []
    for (fr, s), v in chimeric.items():
        keys = list(dict.fromkeys(p["key"] for p in v))
        have = [k in prof for k in keys]
        feas["chimeric scans"] += 1
        feas[f"{sum(have)} of {len(keys)} partners have a clean reference"] += 1
        feas["at least one partner has a reference"] += any(have)
        feas["every partner has a reference"] += all(have)
        d = scans[fr].get(s)
        if len(keys) != 2 or not all(have) or not d or d["nfound"] < 16:
            continue
        O = np.asarray(d["rep"]); sh = d.get("shares", {})
        for a, b in ((keys[0], keys[1]), (keys[1], keys[0])):
            sa, sb = sh.get(a, float("nan")), sh.get(b, float("nan"))
            pa, pb = prof[a][0], prof[b][0]
            naive = cmp_profiles(O, pb)
            if not (sa == sa):
                continue
            est = np.clip(O - sa * O.sum() * pa, 0, None)
            sub = cmp_profiles(est, pb)
            # oracle: 두 기준으로 O 를 맞췄을 때 A 의 몫 (MS1 몫이 리포터 몫을 예측하는가)
            W, *_ = np.linalg.lstsq(np.vstack([pa, pb]).T, O, rcond=None)
            W = np.clip(W, 0, None)
            fit_share = W[0] / W.sum() if W.sum() > 0 else float("nan")
            ms1_share = sa / (sa + sb) if (sa + sb) > 0 else float("nan")
            partner_diff = cmp_profiles(pa, pb)[0]
            On = O / O.sum()
            best = min((cmp_profiles(np.clip(On - al * pa, 0, None), pb)[0], al) for al in np.arange(0, 1.0001, 0.05)
                       if np.clip(On - al * pa, 0, None).sum() > 0)
            to_a = cmp_profiles(O, pa)[0]
            tests.append(dict(fr=fr, scan=s, A=a, B=b, sA=sa, sB=sb, purity=d["purity"],
                              naive_mae=naive[0], naive_r=naive[1], sub_mae=sub[0], sub_r=sub[1],
                              fit_share=fit_share, ms1_share=ms1_share, partner_mae=partner_diff,
                              nrefA=prof[a][1], nrefB=prof[b][1], naive_mae_to_A=to_a,
                              best_mae=best[0], best_alpha=best[1]))

    # ---------------- (B) fr_01 case study
    ours = {}
    with open(OURS, encoding="utf-8") as fh:
        hdr = None
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if hdr is None:
                hdr = f; continue
            r = dict(zip(hdr, f)); ours[int(r["scan"])] = r
    cases = []
    for (fr, s), lst in chim.items():
        if fr != 1:
            continue
        d = scans[1].get(s, {})
        o = ours.get(s)
        Msel = (d.get("sel_mz", 0) - PROTON) * d.get("z", 0)
        tgt = [p for p in lst if any(abs(Msel - p["mass"] - k * C13) <= 20e-6 * p["mass"] for k in (-1, 0, 1, 2))]
        p = tgt[0] if tgt else lst[0]
        oseq, omods = our_form(o["peptide"]) if o and o["peptide"] else ("", {})
        okey = key(oseq, omods) if oseq else ""
        if not o:
            why = "not in our input (charge > 4 or < 20 peaks)" if d else "not in our input"
            cls = "missing"
        elif not tgt:
            cls = "co-isolated only"
        elif okey == p["key"]:
            cls = "agree"
        elif oseq.replace("I", "L") == p["seq"].replace("I", "L"):
            ph_o = sorted(k for k, v in omods.items() if v == "Phospho"); ph_p = sorted(k for k, v in p["mods"].items() if v == "Phospho")
            cls = "phosphosite differs" if len(ph_o) == len(ph_p) and {k: v for k, v in omods.items() if v != "Phospho"} == {k: v for k, v in p["mods"].items() if v != "Phospho"} else "other modification differs"
        elif o["mass_ok"] != "1":
            cls = "our read misses precursor mass"
        else:
            cls = "different sequence, mass consistent"
        unl = (p["mods"].get(0) != "TMTpro") or any(a == "K" and p["mods"].get(i + 1) != "TMTpro" for i, a in enumerate(p["seq"]))
        cov_p = cov_o = expl_p = expl_o = float("nan")
        if "mz" in d:
            cov_p, expl_p = frag_cover(p["seq"], p["mods"], p["z"], d["mz"], d["it"])
            if oseq:
                cov_o, expl_o = frag_cover(oseq, omods, int(o["charge"]), d["mz"], d["it"])
        # 앞·뒤 공통 길이 (다른 서열일 때 어디서 갈라졌나)
        a1, b1 = oseq.replace("I", "L"), p["seq"].replace("I", "L")
        pre = next((i for i in range(min(len(a1), len(b1))) if a1[i] != b1[i]), min(len(a1), len(b1)))
        suf = next((i for i in range(min(len(a1), len(b1))) if a1[-1 - i] != b1[-1 - i]), min(len(a1), len(b1)))
        pd_mz_hit = any(abs(d.get("sel_mz", 0) - p["mono_mz"] - k * C13 / p["z"]) <= 20e-6 * p["mono_mz"] for k in (-1, 0, 1, 2))
        cases.append(dict(scan=s, cls=cls, scan_z=d.get("z", 0), sel_mz=d.get("sel_mz", 0), pd_mz=round(p["mono_mz"], 5),
                          mz_matches_selected=int(pd_mz_hit), charge_mismatch=int(d.get("z", 0) != p["z"]), chim_pep=p["key"], our_pep=okey, length=len(p["seq"]), z=p["z"],
                          n_phos=sum(v == "Phospho" for v in p["mods"].values()), unlabelled=int(unl),
                          ox=int("Oxidation" in p["mods"].values()), chimeric=int(len({q["key"] for q in lst}) > 1),
                          purity=d.get("purity", float("nan")), interf=p["interf"], npk=d.get("npk", 0),
                          our_score=float(o["score"]) if o else float("nan"), our_ppm=float(o["ppm"]) if o and o["ppm"] not in ("", "nan") else float("nan"),
                          cov_chim=cov_p, expl_chim=expl_p, cov_ours=cov_o, expl_ours=expl_o,
                          common_prefix=pre, common_suffix=suf, in_sequest=int(any(q["key"] == p["key"] for q in seqh.get((1, s), [])))))

    # ours-only (fr_01): mass-matched target predictions on scans without any CHIMERYS PSM
    ours_only = [r for sc, r in ours.items() if (1, sc) not in chim and r["mass_ok"] == "1" and r["fasta_match"] == "target"]
    ours_only_seq = sum(1 for sc, r in ours.items() if (1, sc) not in chim and (1, sc) in seqh and r["mass_ok"] == "1" and r["fasta_match"] == "target")

    # ---------------- write
    def dump(name, rows):
        if not rows:
            return
        with open(out / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
    dump("deconv_tests.tsv", tests)
    dump("case_study_fr01.tsv", cases)

    summ = {"feasibility": dict(feas), "n_reference_peptidoforms": len(prof),
            "noise_floor_mae_median": statistics.median([x for x in noise if x == x]) if noise else None,
            "noise_floor_n": len(noise),
            "n_chimeric_scans_all_fractions": len(chimeric)}
    if tests:
        t = [x for x in tests if x["naive_mae"] == x["naive_mae"] and x["sub_mae"] == x["sub_mae"]]
        summ["deconv"] = dict(
            n_tests=len(t), n_scans=len({(x["fr"], x["scan"]) for x in t}),
            naive_mae_median=statistics.median(x["naive_mae"] for x in t),
            sub_mae_median=statistics.median(x["sub_mae"] for x in t),
            improved_frac=sum(x["sub_mae"] < x["naive_mae"] for x in t) / len(t),
            naive_r_median=statistics.median(x["naive_r"] for x in t),
            sub_r_median=statistics.median(x["sub_r"] for x in t),
            partner_mae_median=statistics.median(x["partner_mae"] for x in t),
            best_mae_median=statistics.median(x["best_mae"] for x in t),
            best_alpha_median=statistics.median(x["best_alpha"] for x in t),
            naive_closer_to_A_frac=sum(x["naive_mae_to_A"] < x["naive_mae"] for x in t) / len(t),
            share_corr=float(np.corrcoef([x["fit_share"] for x in t if x["fit_share"] == x["fit_share"] and x["ms1_share"] == x["ms1_share"]],
                                         [x["ms1_share"] for x in t if x["fit_share"] == x["fit_share"] and x["ms1_share"] == x["ms1_share"]])[0, 1]))
        minor = [x for x in t if x["naive_mae_to_A"] < x["naive_mae"]]   # B 가 섞임의 작은 쪽
        if minor:
            summ["deconv_B_is_minor"] = dict(n_tests=len(minor), naive_mae_median=statistics.median(x["naive_mae"] for x in minor),
                                             sub_mae_median=statistics.median(x["sub_mae"] for x in minor),
                                             best_mae_median=statistics.median(x["best_mae"] for x in minor),
                                             improved_frac=sum(x["sub_mae"] < x["naive_mae"] for x in minor) / len(minor))
        big = [x for x in t if x["partner_mae"] >= 0.3]
        if big:
            summ["deconv_partners_differ"] = dict(
                n_tests=len(big), naive_mae_median=statistics.median(x["naive_mae"] for x in big),
                sub_mae_median=statistics.median(x["sub_mae"] for x in big),
                improved_frac=sum(x["sub_mae"] < x["naive_mae"] for x in big) / len(big))
    cc = collections.Counter(c["cls"] for c in cases)
    summ["case_classes_fr01"] = dict(cc.most_common())
    summ["n_chimerys_scans_fr01"] = len(cases)
    summ["ours_only_fr01"] = dict(mass_matched_target_without_chimerys=len(ours_only), of_which_sequest_identified=ours_only_seq)

    def med(rows, k):
        v = [r[k] for r in rows if r[k] == r[k]]
        return round(statistics.median(v), 3) if v else None
    prof_cls = {}
    for cl in cc:
        rows = [c for c in cases if c["cls"] == cl]
        prof_cls[cl] = dict(n=len(rows), length=med(rows, "length"), charge_ge4=round(sum(c["z"] >= 4 for c in rows) / len(rows), 3),
                            n_phos_ge2=round(sum(c["n_phos"] >= 2 for c in rows) / len(rows), 3),
                            unlabelled=round(sum(c["unlabelled"] for c in rows) / len(rows), 3),
                            chimeric=round(sum(c["chimeric"] for c in rows) / len(rows), 3),
                            purity=med(rows, "purity"), npk=med(rows, "npk"), cov_chim=med(rows, "cov_chim"),
                            cov_ours=med(rows, "cov_ours"), expl_chim=med(rows, "expl_chim"), expl_ours=med(rows, "expl_ours"),
                            our_score=med(rows, "our_score"), in_sequest=round(sum(c["in_sequest"] for c in rows) / len(rows), 3))
    summ["class_profiles"] = prof_cls
    co = [c for c in cases if c["cls"] == "co-isolated only"]
    summ["co_isolated_check"] = dict(n=len(co), pd_mz_matches_selected_ion=sum(c["mz_matches_selected"] for c in co),
                                     charge_mismatch=sum(c["charge_mismatch"] for c in co),
                                     both=sum(c["mz_matches_selected"] and c["charge_mismatch"] for c in co))
    (out / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(summ, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
