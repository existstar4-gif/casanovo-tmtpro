# -*- coding: utf-8 -*-
"""fragpipe_compare.py — FragPipe(DB search) 와 우리 de novo 모델을 같은 스펙트럼에서 비교한다.

논문 §4.1 의 [PENDING: 외부 비교군] 을 메우는 스크립트.

내는 것 세 가지:
  1) 동정 규모      FragPipe 1% FDR PSM 수 / 전체 MS2
  2) 라벨원 독립성   우리 정밀도를 원저자 mzid 라벨 대신 FragPipe 라벨로 재채점
  3) 상호 배타성     FragPipe 가 잡은 것 / 우리가 잡은 것 / 둘 다 / 둘 다 실패

⚠️ 회수율(FragPipe 미동정에서 우리가 건지는 비율)은 여기서 내지 않는다.
   우리 예측이 GT 보유 스펙트럼 4,000개 부분집합에만 있기 때문이다.
   그 지표는 미동정 스펙트럼에 대한 별도 추론이 필요하다(`fragpipe_unidentified.py`).

Usage: python scripts/fragpipe_compare.py PXD072211 [PXD054085 ...]
"""
from __future__ import annotations

import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

IN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "fragpipe_in")
EXT = "data/external"

MOD = re.compile(r"\[[^\]]*\]")

# FragPipe 의 반올림 질량 표기 -> 우리 ProForma 토큰
# n[..] 은 N말단(수소 포함 질량), 잔기[..] 는 잔기+수식 질량
NTERM = {"305": "[TMTpro]-", "43": "[Acetyl]-", "44": "[Carbamyl]-"}
RESIDUE = {
    ("K", "432"): "K[TMTpro]",
    ("C", "160"): "C[Carbamidomethyl]",
    ("M", "147"): "M[Oxidation]",
    ("S", "167"): "S[Phospho]",
    ("T", "181"): "T[Phospho]",
    ("Y", "243"): "Y[Phospho]",
    ("N", "115"): "N[Deamidated]",
    ("Q", "129"): "Q[Deamidated]",
}


# 고정 수식 질량 -> 우리 토큰. fragger.params 에서 읽어 적용한다.
FIXED_TOKEN = {
    ("C", 57.02146): "C[Carbamidomethyl]",
    ("K", 304.20715): "K[TMTpro]",
    ("nterm", 304.20715): "[TMTpro]-",
}


def read_fixed(acc):
    """fragger.params 의 add_* 에서 0 이 아닌 고정 수식을 뽑는다.

    ★ FragPipe 는 고정 수식을 'Modified Peptide' 컬럼에 쓰지 않는다.
      (예: add_C_cysteine 이 켜져 있어도 n[305]DQDQGCR 처럼 C 에 표기가 없다.)
      따라서 비교 전에 우리가 직접 되붙여야 한다. 이 단계를 빠뜨리면
      백본은 90% 넘게 일치하는데 펩타이도폼만 25% 로 보이는 가짜 불일치가 생긴다.
    """
    path = os.path.join(IN, acc, "fragger.params")
    res, nterm = {}, 0.0
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.split("#")[0].strip()
        if not line.startswith("add_") or "=" not in line:
            continue
        key, val = [x.strip() for x in line.split("=", 1)]
        try:
            mass = float(val)
        except ValueError:
            continue
        if abs(mass) < 1e-6:
            continue
        if key == "add_Nterm_peptide":
            nterm = mass
        elif key.startswith("add_") and len(key.split("_")) >= 2:
            aa = key.split("_")[1]
            if len(aa) == 1 and aa.isalpha():
                res[aa.upper()] = mass
    return res, nterm


def fp_to_proforma(mod_pep, plain, fixed_res, fixed_nterm):
    """FragPipe 'Modified Peptide' -> 우리 ProForma 표기.

    가변 수식은 [질량] 표기에서, 고정 수식은 fragger.params 에서 가져와 합친다.
    """
    src = mod_pep.strip() or plain.strip()
    out, i, nterm = [], 0, ""
    seen_var_nterm = False
    while i < len(src):
        ch = src[i]
        if ch == "n" and i + 1 < len(src) and src[i + 1] == "[":
            j = src.index("]", i)
            nterm = NTERM.get(src[i + 2:j].split(".")[0], "")
            seen_var_nterm = True
            i = j + 1
            continue
        if ch.isalpha():
            if i + 1 < len(src) and src[i + 1] == "[":
                j = src.index("]", i + 1)
                m = src[i + 2:j].split(".")[0]
                out.append(RESIDUE.get((ch, m), ch))   # 가변 수식
                i = j + 1
                continue
            # 표기가 없는 잔기 — 고정 수식이 걸려 있으면 되붙인다
            fm = fixed_res.get(ch)
            tok = FIXED_TOKEN.get((ch, round(fm, 5))) if fm else None
            out.append(tok or ch)
        i += 1
    if not seen_var_nterm and fixed_nterm:
        nterm = FIXED_TOKEN.get(("nterm", round(fixed_nterm, 5)), "")
    return nterm + "".join(out)


def bare(s):
    return MOD.sub("", s).replace("-", "").strip().replace("I", "L")


def peptidoform(s):
    return s.strip().replace("I", "L")


def load_fragpipe(acc):
    """{(stem, scan): (proforma, qvalue)}"""
    fixed_res, fixed_nterm = read_fixed(acc)
    print(f"  [{acc}] 고정 수식: " +
          ", ".join(f"{k}+{v:.5f}" for k, v in sorted(fixed_res.items())) +
          (f", N-term+{fixed_nterm:.5f}" if fixed_nterm else ", N-term 없음(가변)"))
    path = os.path.join(IN, acc, "psm.tsv")
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        hdr = {k: i for i, k in enumerate(fh.readline().rstrip("\n").split("\t"))}
        si, pi, mi = hdr["Spectrum"], hdr["Peptide"], hdr.get("Modified Peptide")
        qi = hdr.get("Qvalue")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) <= si:
                continue
            parts = c[si].split(".")
            if len(parts) < 4:
                continue
            try:
                scan = int(parts[-3])
            except ValueError:
                continue
            stem = ".".join(parts[:-3])
            mp = c[mi] if mi is not None and len(c) > mi else ""
            try:
                q = float(c[qi]) if qi is not None and len(c) > qi else 0.0
            except ValueError:
                q = 0.0
            out[(stem, scan)] = (fp_to_proforma(mp, c[pi], fixed_res, fixed_nterm), q)
    return out


def load_ours(acc):
    """{(stem, scan): proforma} — sub.mgf 순서로 index -> scan 복원."""
    d = f"{EXT}/{acc}"
    scans, stems = [], []
    for line in open(f"{d}/sub.mgf", encoding="utf-8"):
        if line.startswith("TITLE="):
            t = line.strip()[6:]
            st, sc = t.rsplit(":scan:", 1)
            stems.append(st)
            scans.append(int(sc))
    raw, hdr = {}, None
    for line in open(f"{d}/ext.mztab", encoding="utf-8", errors="replace"):
        if line.startswith("PSH"):
            hdr = line.rstrip("\n").split("\t")
            continue
        if not line.startswith("PSM") or not hdr:
            continue
        c = dict(zip(hdr, line.rstrip("\n").split("\t")))
        m = re.search(r"index=(\d+)", c.get("spectra_ref", ""))
        if m:
            raw[int(m.group(1))] = (
                c.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence")
                or c.get("sequence", ""))
    return {(stems[i], scans[i]): raw.get(i, "") for i in range(len(scans))}


def load_gt(acc):
    gt = {}
    for line in open(f"{EXT}/{acc}/gt.tsv", encoding="utf-8").read().splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 5:
            gt[(c[0], int(c[1]))] = c[4]
    return gt


def run(acc, total_ms2=None):
    fp = load_fragpipe(acc)
    ours = load_ours(acc)
    gt = load_gt(acc)

    print(f"\n{'='*70}\n{acc}\n{'='*70}")
    print(f"  FragPipe 1% FDR PSM      {len(fp):,}")
    if total_ms2:
        print(f"  전체 MS2                 {total_ms2:,}   동정률 {100*len(fp)/total_ms2:.1f}%")
    print(f"  우리 예측 대상(부분집합)  {len(ours):,}")

    # 우리 평가 부분집합 안에서 FragPipe 도 동정한 것
    both = [k for k in ours if k in fp]
    print(f"  ↳ 그중 FragPipe 도 동정   {len(both):,}  ({100*len(both)/len(ours):.1f}%)")
    if not both:
        return

    # (2) 라벨원 독립성 — 같은 스펙트럼을 두 라벨로 각각 채점
    a_gt = a_fp = agree = 0
    n_gt = 0
    for k in both:
        p = ours.get(k, "")
        if not p:
            continue
        g, f = gt.get(k), fp[k][0]
        if g:
            n_gt += 1
            a_gt += peptidoform(p) == peptidoform(g)
        a_fp += peptidoform(p) == peptidoform(f)
        if g:
            agree += peptidoform(g) == peptidoform(f)
    print(f"\n  ── 라벨원 독립성 (두 엔진이 모두 동정한 {len(both):,} 스펙트럼) ──")
    if n_gt:
        print(f"     원저자 mzid 라벨로 채점한 우리 정밀도   {100*a_gt/n_gt:.1f}%  (n={n_gt:,})")
    print(f"     FragPipe 라벨로 채점한 우리 정밀도       {100*a_fp/len(both):.1f}%")
    if n_gt:
        print(f"     두 엔진 라벨끼리의 일치                 {100*agree/n_gt:.1f}%")

    # (3) 상호 배타성 — 백본 기준
    bb = sum(1 for k in both if ours.get(k) and bare(ours[k]) == bare(fp[k][0]))
    print(f"     백본만 비교하면                         {100*bb/len(both):.1f}%")


if __name__ == "__main__":
    TOTAL = {"PXD072211": 163858, "PXD054085": 692310, "PXD063976": 262311}
    for a in (sys.argv[1:] or ["PXD072211", "PXD054085", "PXD063976"]):
        try:
            run(a, TOTAL.get(a))
        except FileNotFoundError as exc:
            print(f"\n{a}: 건너뜀 — {exc}")
