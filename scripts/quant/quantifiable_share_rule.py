# -*- coding: utf-8 -*-
"""quantifiable_share_rule.py — '정량 가능한 인산화 펩타이드' 를 펩타이드별 분리창 몫으로 다시 센다 (2026-10-21, 사용자 요청).

왜: quantifiable_same_rule.py(10-07) 는 PSM 의 순도로 **스캔 순도**(기기가 고른 전구체의 몫)를 썼다.
CHIMERYS 는 같이 분리된 다른 전구체도 동정하는데(Study 2 fr_01 CHIMERYS 스캔의 15.8%), 그 펩타이드에는 스캔 순도가
맞지 않는다. 리포터의 대부분이 다른 펩타이드 것일 수 있다. 그래서 펩타이드 **자신의** 몫을 잰다.

새 규칙 (PSM 정량 가능):
  ① TMT 표지 — N말단과 모든 K 에 TMTpro
  ② 자기 몫 ≥ MIN_SHARE — 그 스캔의 master MS1 분리창 안에서 이 펩타이드의 동위원소(M..M+5) 세기 / 창 안 총세기
     (core.reporters.purity 를 펩타이드의 이론 단일동위원소 m/z·전하로 부른다)
  ③ 값>0 채널 ≥ MIN_CH (스캔 단위 리포터, *.rep.tsv)
  ②가 0.5 이상이면 한 스캔에서 둘이 동시에 통과할 수 없으므로 '한 스캔 한 정량' 은 따로 두지 않는다.
펩타이드(bare, I=L) 정량 가능 ⇔ 정량 가능한 PSM 이 하나라도 있다.
입력은 10-07 과 같다: PD = Confidence High 인산화 PSM, 우리 = decoy 매칭 FDR 1% 문턱 이상 · 참조 공간 안 인산화 PSM.
옛 규칙(스캔 순도 ≥ 0.5, 채널 ≥ 8)도 같은 표에 낸다.

    python scripts/quant/quantifiable_share_rule.py --study-dir out/quant/study2 --pd-base "data/pd/study2_phospho_fr_" \
        --fasta <mouse.fasta> --cut 0.1330 --label "Study 2"
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quantifiable_same_rule as qs            # noqa: E402  (read_rep, bare)
import count_matched_fdr as cm                 # noqa: E402
from count_matched_fdr import pp, normalize    # noqa: E402
import chimeric_case_study as cs               # noqa: E402  (pd_form, our_form, mass_of, rp)

MZDIR = Path(r"data/mzml_tmt")
FILEID_MAP: dict = {}  # --fileid-map: InputFiles.txt 가 없는 export 용 (infer_pd_fileid.py)
ALL_PEPTIDES = False   # --all-peptides: global proteome 용, 인산화 거름을 끈다 (2026-11-17)


def labelled(seq, mods):
    return mods.get(0) == "TMTpro" and all(mods.get(i + 1) == "TMTpro" for i, a in enumerate(seq) if a == "K")


def read_pd(path):
    """PD PSMs -> [dict(stem, scan, bare, mz, z, lab)] — Confidence High · 인산화만"""
    fid = pp.load_inputfiles_map(path) or FILEID_MAP
    out = []
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        hdr = [pp.clean_field(c) for c in fh.readline().rstrip("\n").split("\t")]
        ix = {n: i for i, n in enumerate(hdr)}
        for line in fh:
            c = [pp.clean_field(x) for x in line.rstrip("\n").split("\t")]
            if c[ix["Confidence"]] != "High" or (not ALL_PEPTIDES and "Phospho" not in c[ix["Modifications"]]):
                continue
            seq, mods = cs.pd_form(c[ix["Annotated Sequence"]], c[ix["Modifications"]])
            z = int(c[ix["Charge"]]); mh = float(c[ix["Theo. MH+ [Da]"]])
            out.append(dict(stem=fid.get(c[ix["File ID"]]), scan=int(c[ix["First Scan"]]), bare=normalize(seq),
                            mz=(mh - cs.PROTON + z * cs.PROTON) / z, z=z, lab=labelled(seq, mods),
                            _seq=seq, _mods=mods, nphos=sum(v == "Phospho" for v in mods.values())))
    return out


def read_ours(paths, cut):
    out = []
    for p in paths:
        seen = False
        for line in io.open(p, encoding="utf-8"):
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[:2] == ["stem", "scan"]:
                seen = True; continue
            if not seen or (not ALL_PEPTIDES and "Phospho" not in c[2]):
                continue
            try:
                sc = float(c[4]); z = int(c[6])
            except ValueError:
                continue
            if sc < cut:
                continue
            seq, mods = cs.our_form(c[2])
            out.append(dict(stem=c[0], scan=int(c[1]), bare=qs.bare(c[2]), mz=(cs.mass_of(seq, mods) + z * cs.PROTON) / z,
                            z=z, lab=labelled(seq, mods)))
    return out


def own_shares(stem, want):
    """want = {scan: [(id, mz, z)]} -> {id: share}. master MS1(spectrumRef) 의 분리창으로 잰다."""
    from pyteomics import mzml
    rp = cs.rp
    got, ms1, order, last = {}, {}, [], None
    with mzml.read(str(MZDIR / (stem + ".mzML"))) as rd:
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
            items = want.get(scan)
            if not items:
                continue
            prec = sp["precursorList"]["precursor"][0]
            sel = prec["selectedIonList"]["selectedIon"][0]
            smz = float(sel.get("selected ion m/z", 0.0))
            win = prec.get("isolationWindow", {})
            center = float(win.get("isolation window target m/z", smz) or smz)
            lo = float(win.get("isolation window lower offset", 0.7) or 0.7)
            hi = float(win.get("isolation window upper offset", lo) or lo)
            ref = rp.scan_of(prec.get("spectrumRef")) or last
            for i, mz, z in items:
                got[i] = rp.purity(*ms1[ref], mz, z, center, lo, hi) if ref in ms1 else float("nan")
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", required=True)
    ap.add_argument("--pd-base", required=True, help="PD export 경로에서 '_PSMs.txt' / '-(1)_PSMs.txt' 앞부분")
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--cut", type=float, required=True)
    ap.add_argument("--min-share", type=float, default=0.5)
    ap.add_argument("--min-ch", type=int, default=8)
    ap.add_argument("--label", default="study")
    ap.add_argument("--json")
    ap.add_argument("--pd-in-reference", action="store_true",
                    help="PD 펩타이드도 우리와 같은 참조 공간(--fasta 의 트립신 조각)에 있는 것만 센다 (2026-10-22: PD 는 TrEMBL 포함 DB 로 검색했다)")
    ap.add_argument("--fileid-map", default=None, help="File ID -> stem JSON (InputFiles.txt 가 없을 때)")
    ap.add_argument("--all-peptides", action="store_true",
                    help="인산화 거름 없이 모든 펩타이드를 센다 (global proteome, 2026-11-17)")
    a = ap.parse_args()
    global ALL_PEPTIDES, FILEID_MAP
    ALL_PEPTIDES = a.all_peptides
    if a.fileid_map:
        FILEID_MAP = json.load(open(a.fileid_map, encoding="utf-8"))

    rep = qs.read_rep(sorted(glob.glob(os.path.join(a.study_dir, "*.rep.tsv"))))
    ours = read_ours(sorted(glob.glob(os.path.join(a.study_dir, "*_psm.tsv"))), a.cut)
    tgt, _dec = cm.target_decoy(a.fasta, {r["bare"] for r in ours})
    ours = [r for r in ours if r["bare"] in tgt]
    sets = {"Sequest HT": read_pd(a.pd_base + "_PSMs.txt"), "CHIMERYS": read_pd(a.pd_base + "-(1)_PSMs.txt"), "Ours": ours}
    if a.pd_in_reference:
        ptgt, _ = cm.target_decoy(a.fasta, {p["bare"] for n in ("Sequest HT", "CHIMERYS") for p in sets[n]})
        for n in ("Sequest HT", "CHIMERYS"):
            sets[n] = [p for p in sets[n] if p["bare"] in ptgt]

    # 펩타이드별 몫: 분획별로 한 번씩 mzML 을 훑는다
    want = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, ps in sets.items():
        for i, p in enumerate(ps):
            p["id"] = (name, i)
            want[p["stem"]][p["scan"]].append((p["id"], p["mz"], p["z"]))
    share = {}
    for k, stem in enumerate(sorted(want), 1):
        share.update(own_shares(stem, want[stem]))
        print(f"  [{k}/{len(want)}] {stem}", flush=True)

    res = {}
    for name, ps in sets.items():
        allp, old, new, lab_ok = set(), set(), set(), set()
        why = collections.Counter()
        for p in ps:
            allp.add(p["bare"])
            v = rep.get((p["stem"], p["scan"]))
            if v is None:
                why["no reporter row"] += 1; continue
            pur, nch = v
            if pur >= 0.5 and nch >= a.min_ch:
                old.add(p["bare"])
            s = share.get(p["id"], float("nan"))
            if not p["lab"]:
                why["not fully TMT-labelled"] += 1
            elif not (s >= a.min_share):
                why[f"own share < {a.min_share}"] += 1
            elif nch < a.min_ch:
                why[f"< {a.min_ch} channels"] += 1
            else:
                new.add(p["bare"])
        res[name] = dict(identified=len(allp), old_rule=len(old), new_rule=len(new), psm_fail_reasons=dict(why),
                         old_minus_new=len(old - new), new_only=len(new - old))
    print("=" * 90)
    print(f"{a.label}: quantifiable {'peptides' if ALL_PEPTIDES else 'phosphopeptides'} (bare, I=L)")
    print(f"  old rule = scan purity >= 0.5 & channels >= {a.min_ch}  |  new rule = TMT-labelled & own share >= {a.min_share} & channels >= {a.min_ch}")
    print("%-12s %10s %10s %10s %10s" % ("method", "identified", "old rule", "new rule", "change"))
    for name, r in res.items():
        print("%-12s %10s %10s %10s %9.1f%%" % (name, f"{r['identified']:,}", f"{r['old_rule']:,}", f"{r['new_rule']:,}",
                                               100 * (r["new_rule"] / r["old_rule"] - 1) if r["old_rule"] else float("nan")))
    for rule in ("old_rule", "new_rule"):
        o = res["Ours"][rule]
        print(f"  {rule}: ours vs Sequest {100 * (o / res['Sequest HT'][rule] - 1):+.1f}%  vs CHIMERYS {100 * (o / res['CHIMERYS'][rule] - 1):+.1f}%")
    for name, r in res.items():
        print(f"  {name} PSM fail reasons (new rule): {r['psm_fail_reasons']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
