# -*- coding: utf-8 -*-
"""phospho_decompose.py — 외부 인산화 데이터에서 오류를 백본/위치결정으로 분해한다.

기존 `phospho_eval.py` 의 T1~T4 계층은 n=79 규모였고, T2/T3/T4 가 항상 같아
"검출 recall 100%" 라는 산술 항등식을 산출했다(철회됨).

여기서는 그 항등식을 피해 **실제로 물을 수 있는 질문 하나**만 본다:
    백본 서열을 맞혔을 때, 인산화 위치까지 맞히는 비율은?
이는 전구체 질량으로 자동 보장되지 않는다 — 같은 질량의 배치 후보가 여럿이기 때문이다.
분모를 "배치 후보가 2개 이상인 경우"로 좁히면 항등식 성분이 완전히 제거된다.

Usage: python scripts/phospho_decompose.py PXD063976 [PXD054497 ...]
"""
from __future__ import annotations

import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

MOD = re.compile(r"\[[^\]]*\]")
PH_SITE = re.compile(r"([STYRK])\[Phospho\]")


def backbone(s):
    """수식 전부 제거 + I/L 동일시."""
    return MOD.sub("", s).replace("-", "").strip().replace("I", "L")


def n_phos(s):
    return s.count("[Phospho]")


def sites(s):
    """인산화된 잔기의 백본 상 0-based 위치 집합."""
    out, idx = set(), 0
    i = 0
    while i < len(s):
        if s[i] == "[":
            j = s.index("]", i)
            if s[i:j + 1] == "[Phospho]":
                out.add(idx - 1)
            i = j + 1
            continue
        if s[i].isalpha():
            idx += 1
        i += 1
    return out


def load_mztab(path):
    out, hdr = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("PSH"):
            hdr = line.rstrip("\n").split("\t")
            continue
        if not line.startswith("PSM") or not hdr:
            continue
        c = dict(zip(hdr, line.rstrip("\n").split("\t")))
        m = (re.search(r"index=(\d+)", c.get("spectra_ref", ""))
             or re.search(r"scan=(\d+)", c.get("spectra_ref", "")))
        if m:
            out[int(m.group(1))] = (
                c.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence")
                or c.get("sequence", ""))
    return out


def run(acc):
    d = f"data/external/{acc}"
    # ⚠️ 정답 키는 (파일 stem, scan). scan 단독으로 잡으면 파일 간 scan 번호가 겹쳐
    #    라벨이 덮인다. PXD054497 은 stem 2개에 scan 228건이 겹쳐 2,300건 중 224건이
    #    오염돼 있었다 (2026-08-28 — score_external 에서 고친 것과 같은 버그).
    gt = {}
    for line in open(f"{d}/gt.tsv", encoding="utf-8").read().splitlines()[1:]:
        c = line.split(chr(9))
        if len(c) >= 5:
            gt[(c[0], int(c[1]))] = c[4]
    scans = []
    for l in open(f"{d}/sub.mgf", encoding="utf-8"):
        if l.startswith("TITLE="):
            st, sc = l.strip()[6:].rsplit(":scan:", 1)
            scans.append((st, int(sc)))
    # ext.mztab 과 ext_knap.mztab 은 둘 다 knapsack ON 이다(2026-08-31 확인, 예측 차이 0).
    # PXD067164 는 ext_knap 이름으로만 있으므로 없으면 그쪽을 쓴다. OFF 는 ext_noknap.mztab.
    mzt = f"{d}/ext.mztab"
    if not os.path.exists(mzt):
        mzt = f"{d}/ext_knap.mztab"
    raw = load_mztab(mzt)

    n = bb_ok = exact = 0
    ph_gt = 0
    bb_ok_ph = site_ok = count_ok = 0
    amb_n = amb_ok = 0          # 배치 후보 2개 이상인 경우만
    for i, sc in enumerate(scans):
        g = gt.get(sc)
        if not g:
            continue
        p = raw.get(i, raw.get(sc[1], ""))
        n += 1
        if not p:
            continue
        gb, pb = backbone(g), backbone(p)
        if gb != pb:
            continue
        bb_ok += 1
        if p.replace("I", "L").strip() == g.replace("I", "L").strip():
            exact += 1
        if n_phos(g) == 0:
            continue
        ph_gt += 1
        bb_ok_ph += 1
        if n_phos(p) == n_phos(g):
            count_ok += 1
        same = sites(p) == sites(g)
        site_ok += same
        # 백본 안에서 인산화 가능한 잔기(S/T/Y) 수가 인산화 개수보다 많으면 '모호'
        candidates = sum(1 for ch in gb if ch in "STY")
        if candidates > n_phos(g):
            amb_n += 1
            amb_ok += same

    print(f"\n=== {acc} ===")
    print(f"  평가 스펙트럼            {n:,}")
    print(f"  백본 일치                {bb_ok:,}  ({100*bb_ok/n:.1f}%)")
    print(f"  펩타이도폼 완전 일치     {exact:,}  ({100*exact/n:.1f}%)")
    if ph_gt:
        print(f"  ── 백본을 맞힌 인산화 펩타이드 {bb_ok_ph:,} 건 안에서 ──")
        print(f"     인산화 개수 일치      {count_ok:,}  ({100*count_ok/bb_ok_ph:.1f}%)")
        print(f"     인산화 위치 일치      {site_ok:,}  ({100*site_ok/bb_ok_ph:.1f}%)")
        print(f"     ↳ 위치 오류만         {bb_ok_ph-site_ok:,}  ({100*(bb_ok_ph-site_ok)/bb_ok_ph:.1f}%)")
        if amb_n:
            print(f"  ★ 배치 후보 2개 이상(S/T/Y 여유 있음)만: "
                  f"{amb_ok:,}/{amb_n:,} = {100*amb_ok/amb_n:.1f}%")
            print(f"     ← 질량으로 자동 결정되지 않는 경우만 남긴 값. 항등식 성분 없음.")
    return dict(n=n, bb=bb_ok, exact=exact, ph=bb_ok_ph,
                site=site_ok, amb_n=amb_n, amb_ok=amb_ok)


if __name__ == "__main__":
    accs = sys.argv[1:] or ["PXD063976", "PXD054497"]
    for a in accs:
        run(a)
