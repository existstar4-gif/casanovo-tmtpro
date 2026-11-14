# -*- coding: utf-8 -*-
"""quant_loss_breakdown.py — Fig. 4A 에서 4B 로 줄어드는 인산화 펩타이드를 펩타이드 단위로 원인별로 나눈다 (2026-11-13, 사용자 요청).

Fig. 4A 집합(overlap4_phospho 와 같은 규칙)의 펩타이드마다, 정량 가능한 PSM 이 하나도 없으면 원인 하나를 붙인다:
  label    — 표지가 완전한 PSM 이 없다 (Casanovo-TMT 만 해당: K 미표지 / N말단 아세틸 / N말단 미표지로 더 나눈다)
  share    — 표지 완전 PSM 은 있으나 모두 분리창 자기 몫 < 0.5 (같이 분리된 전구체)
  channels — 몫 ≥ 0.5 인 표지 완전 PSM 은 있으나 모두 리포터 채널 < 8 (또는 리포터 행 없음)
규칙은 quantifiable_share_rule.py 와 같다 (PSM: 완전 표지 & 자기 몫 ≥ 0.5 & 채널 ≥ 8).

    python scripts/quant/quant_loss_breakdown.py
출력: out/quant/validation/quant_loss_<study>.json
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quantifiable_share_rule as qr          # noqa: E402
import quantifiable_same_rule as qs           # noqa: E402
import fragpipe_phospho_compare as fpc        # noqa: E402
import count_matched_fdr as cm                # noqa: E402
import overlap4_phospho as ov                 # noqa: E402
import chimeric_case_study as cs              # noqa: E402
from rescued_pick import FA, P, BASES         # noqa: E402

PDB = "data/pd/"
FP_ST = {"study1": "s1", "study2": "s2", "study3": "s3"}
CUT = {}   # 인산화 문턱 = split_phospho_pfdr_lab_<study>.log


def cut_of(study):
    import re
    t = open(f"out/quant/split_phospho_pfdr_lab_{study}.log", encoding="utf-8").read()
    return float(re.search(r"인산화 문턱 = ([0-9.]+)", t).group(1))


def ours_psms(study, cut):
    rows = qr.read_ours(sorted(glob.glob(f"out/quant/{study}/*_psm.tsv")), cut)
    tgt, _ = cm.target_decoy(FA, {r["bare"] for r in rows})
    out = []
    for r in rows:
        if r["bare"] not in tgt:
            continue
        out.append(r)
    return out


def label_kind(study):
    """Casanovo-TMT PSM 의 미표지 종류: (stem, scan) -> 'nterm_unlab' | 'nterm_acetyl' | 'k_unlab' (완전 표지면 없음)"""
    kind = {}
    for p in sorted(glob.glob(f"out/quant/{study}/*_psm.tsv")):
        seen = False
        for line in open(p, encoding="utf-8"):
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[:2] == ["stem", "scan"]:
                seen = True; continue
            if not seen or "Phospho" not in c[2]:
                continue
            seq, mods = cs.our_form(c[2])
            if mods.get(0) != "TMTpro":
                k = "nterm_acetyl" if mods.get(0) == "Acetyl" else "nterm_unlab"
            elif not all(mods.get(i + 1) == "TMTpro" for i, a in enumerate(seq) if a == "K"):
                k = "k_unlab"
            else:
                continue
            kind[(c[0], int(c[1]))] = k
    return kind


def main():
    for study in ("study1", "study2", "study3"):
        b = BASES[study]
        cut = cut_of(study)
        sets4a = {"SEQUEST HT": ov.cps.read_pd(P + b + "_PeptideGroups.txt")[0],
                  "CHIMERYS": ov.cps.read_pd(P + b + "-(1)_PeptideGroups.txt")[0],
                  "FragPipe": ov.fragpipe(study), "Casanovo-TMT": ov.ours(study)}
        psms = {"SEQUEST HT": qr.read_pd(PDB + b + "_PSMs.txt"),
                "CHIMERYS": qr.read_pd(PDB + b + "-(1)_PSMs.txt"),
                "FragPipe": [p for p in fpc.read_default(FP_ST[study]) if p["phos"]],
                "Casanovo-TMT": ours_psms(study, cut)}
        kind = label_kind(study)
        rep = qs.read_rep(sorted(glob.glob(f"out/quant/{study}/*.rep.tsv")))
        want = collections.defaultdict(lambda: collections.defaultdict(list))
        for name, ps in psms.items():
            for i, p in enumerate(ps):
                p["id"] = (name, i)
                want[p["stem"]][p["scan"]].append((p["id"], p["mz"], p["z"]))
        share = {}
        for k, stem in enumerate(sorted(want), 1):
            share.update(qr.own_shares(stem, want[stem]))
            print(f"  {study} [{k}/{len(want)}] {stem}", flush=True)
        res = {}
        for name, ps in psms.items():
            by = collections.defaultdict(list)
            for p in ps:
                by[p["bare"]].append(p)
            cnt = collections.Counter()
            for pep in sets4a[name]:
                ok_lab = share_ok = quant = False
                lk = collections.Counter()
                for p in by.get(pep, []):
                    if not p["lab"]:
                        lk[kind.get((p["stem"], p["scan"]), "other")] += 1
                        continue
                    ok_lab = True
                    if not (share.get(p["id"], float("nan")) >= 0.5):
                        continue
                    share_ok = True
                    v = rep.get((p["stem"], p["scan"]))
                    if v is not None and v[1] >= 8:
                        quant = True
                if quant:
                    cnt["quantifiable"] += 1
                elif not by.get(pep):
                    cnt["no PSM above threshold"] += 1
                elif not ok_lab:
                    cnt["label"] += 1
                    cnt["label:" + lk.most_common(1)[0][0]] += 1
                elif not share_ok:
                    cnt["share"] += 1
                else:
                    cnt["channels"] += 1
            res[name] = dict(identified=len(sets4a[name]), **cnt)
            print(study, name, dict(res[name]), flush=True)
        json.dump(dict(study=study, cut=cut, methods=res),
                  open(f"out/quant/validation/quant_loss_{study}.json", "w", encoding="utf-8"), indent=2)
    print("QUANT_LOSS_DONE", flush=True)


if __name__ == "__main__":
    main()
