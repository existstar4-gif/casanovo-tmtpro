# -*- coding: utf-8 -*-
"""cluster_ci.py — 개발셋 헤드라인 정밀도에 펩타이드 클러스터 부트스트랩 CI 를 붙인다.

★ 왜 필요한가
  개발셋은 겉보기 4,000 스펙트럼이지만 고유 펩타이드가 **823개**뿐이고, 최다 펩타이드
  하나가 171건(4.3%)을 차지한다(`AUDIT_2026-08-15.md` §3; 891·167 은 구판 오기).
  스펙트럼을 독립 표본으로 보면 반복 측정만큼 CI 를 과소평가한다.
  `PREREG_holdout.md` 는 클러스터 CI 병기를 스스로 의무화했고, 원고 초안에는
  신뢰구간 표기가 0건이었다(2026-08-29 적대 검토 지적).

  여기서는 **펩타이드를 재표집 단위**로 하는 클러스터 부트스트랩을 쓴다. 같은 펩타이드의
  스펙트럼은 함께 뽑히거나 함께 빠지므로, 반복 측정이 만드는 가짜 정밀도가 제거된다.

  개발셋 정답 파일에는 study/stem 열이 없어 스터디 클러스터는 계산할 수 없다.
  감사 문서의 스터디 클러스터 구간(구판 값 기준)이 펩타이드 클러스터보다 약 1.7배
  넓었으므로, 아래 구간은 **보수적이지 않은 쪽**임을 논문에 밝혀야 한다.

  ★ 2026-08-31 추가: `--ext` 로 외부 예치물 6건에도 같은 CI 를 붙인다. 원고의 Figure 3A 가
  오차막대 없는 막대 6개였고, 그 값들의 차이가 CI 안인지 밖인지 그림만 보고 알 수 없었다.

사용:
  python scripts/cluster_ci.py [mztab]        기본 data/eval/final_knap.mztab
  python scripts/cluster_ci.py --ext          외부 예치물 6건
"""
from __future__ import annotations

import os
import random
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

GT = "data/eval/sample_new_gt.tsv"
DEFAULT_MZTAB = "data/eval/final_knap.mztab"
MOD = re.compile(r"\[[^\]]*\]")
NL = chr(10)
TAB = chr(9)
B = 10000
SEED = 20260829


def bare(s):
    return MOD.sub("", s).replace("-", "").strip().upper()


def full(s):
    return s.replace("I", "L").strip()


def load_mztab(path):
    out, hdr = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("PSH"):
            hdr = line.rstrip(NL).split(TAB)
            continue
        if not line.startswith("PSM") or not hdr:
            continue
        c = dict(zip(hdr, line.rstrip(NL).split(TAB)))
        m = re.search(r"index=(\d+)", c.get("spectra_ref", ""))
        if m:
            out[int(m.group(1))] = (
                c.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence")
                or c.get("sequence", ""))
    return out


def boot(clusters, rng, b=B):
    """펩타이드 클러스터 재표집. clusters = [(맞은 수, 전체 수), …]"""
    if not clusters:
        return (float("nan"),) * 2
    k = len(clusters)
    out = []
    for _ in range(b):
        ok = tot = 0
        for _ in range(k):
            a, n = clusters[rng.randrange(k)]
            ok += a
            tot += n
        if tot:
            out.append(100 * ok / tot)
    out.sort()
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out))])


def naive(k, n, z=1.96):
    """스펙트럼을 독립으로 본 Wilson 구간 — 비교용(과소평가)."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (100 * (c - h), 100 * (c + h))


EXT = "data/external"
SIX = ["PXD054085", "PXD065357", "PXD072211", "PXD063976", "PXD054497", "PXD067164"]


def ext_clusters(acc):
    """외부 예치물 하나의 (맞은 수, 전체 수) 펩타이드 클러스터 목록."""
    d = f"{EXT}/{acc}"
    mzt = f"{d}/ext.mztab"
    if not os.path.exists(mzt):
        mzt = f"{d}/ext_knap.mztab"
    if not (os.path.exists(mzt) and os.path.exists(f"{d}/sub.mgf")):
        return None
    gt = {}
    for line in open(f"{d}/gt.tsv", encoding="utf-8").read().splitlines()[1:]:
        c = line.split(TAB)
        if len(c) >= 5:
            gt[(c[0], int(c[1]))] = c[4]
    keys = []
    for line in open(f"{d}/sub.mgf", encoding="utf-8"):
        if line.startswith("TITLE="):
            st, sc = line.strip()[6:].rsplit(":scan:", 1)
            keys.append((st, int(sc)))
    pred = load_mztab(mzt)
    agg = defaultdict(lambda: [0, 0])
    for i, k in enumerate(keys):
        g = gt.get(k)
        if not g:
            continue
        p = pred.get(i, "")
        pep = bare(g).replace("I", "L")
        agg[pep][0] += bool(p) and full(p) == full(g)
        agg[pep][1] += 1
    return [(a, t) for a, t in agg.values()]


def ext_ci(acc, seed=SEED):
    """(정밀도, lo, hi, n, 고유펩) — Figure 3A 오차막대용."""
    cl = ext_clusters(acc)
    if not cl:
        return None
    n = sum(t for _, t in cl)
    k = sum(a for a, _ in cl)
    lo, hi = boot(cl, random.Random(seed))
    return (100 * k / n, lo, hi, n, len(cl))


def main_ext():
    """표를 찍고, Figure 3A 오차막대가 읽을 캐시(out/ext_ci.json)를 남긴다.
    부트스트랩이 느려서(예치물당 수천 클러스터 x 10,000회) 그림 생성 때마다 돌리지 않는다."""
    import json
    print(f"외부 예치물 — 펩타이드 클러스터 부트스트랩 {B:,}회, seed={SEED}" + NL)
    print(f"{'예치물':<12}{'n':>7}{'고유펩':>8}{'정밀도':>9}"
          f"{'  클러스터 95% CI':<24}{'  (독립가정)':<20}")
    cache = {}
    for acc in SIX:
        r = ext_ci(acc)
        if not r:
            continue
        prec, lo, hi, n, k = r
        nlo, nhi = naive(round(prec * n / 100), n)
        print(f"{acc:<12}{n:>7,}{k:>8,}{prec:>8.1f}%"
              f"   {lo:5.1f} – {hi:5.1f}  (±{(hi-lo)/2:.1f}%p)"
              f"      {nlo:5.1f} – {nhi:5.1f}  (±{(nhi-nlo)/2:.1f}%p)")
        cache[acc] = {"precision": prec, "lo": lo, "hi": hi, "n": n, "peptides": k}
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "out", "ext_ci.json")
    json.dump(cache, open(out, "w", encoding="utf-8"), indent=1)
    print(NL + f"캐시 기록: {out}  (make_figures.fig3 가 오차막대로 읽는다)")


def main():
    if "--ext" in sys.argv:
        return main_ext()
    mztab = next((a for a in sys.argv[1:] if not a.startswith("-")), DEFAULT_MZTAB)
    pred = load_mztab(mztab)
    rows = [l.split(TAB) for l in open(GT, encoding="utf-8").read().splitlines()[1:]
            if l.strip()]
    rows = [r for r in rows if len(r) >= 3]

    rec = []
    for r in rows:
        idx, cls, seq = int(r[0]), r[1], r[2]
        p = pred.get(idx, "")
        rec.append({"cls": cls, "pep": bare(seq).replace("I", "L"),
                    "ok": bool(p) and full(p) == full(seq)})

    peps = {r["pep"] for r in rec}
    top = max(peps, key=lambda q: sum(1 for r in rec if r["pep"] == q))
    ntop = sum(1 for r in rec if r["pep"] == top)
    print(f"표본 구조: 스펙트럼 {len(rec):,}  고유 펩타이드 {len(peps):,}  "
          f"최다 펩타이드 {ntop:,}건 ({100*ntop/len(rec):.1f}%)")
    print(f"부트스트랩 {B:,}회, seed={SEED}, 재표집 단위 = 펩타이드{NL}")

    rng = random.Random(SEED)
    print(f"{'집합':<12}{'n':>7}{'고유펩':>8}{'정밀도':>9}"
          f"{'  펩타이드 클러스터 95% CI':<26}{'  (독립가정 CI)':<20}")
    for label, sub in (("전체", rec),
                       ("TMT", [r for r in rec if r["cls"] == "tmt"]),
                       ("label-free", [r for r in rec if r["cls"] == "naked"])):
        n = len(sub)
        k = sum(r["ok"] for r in sub)
        agg = defaultdict(lambda: [0, 0])
        for r in sub:
            agg[r["pep"]][0] += r["ok"]
            agg[r["pep"]][1] += 1
        cl = [(a, t) for a, t in agg.values()]
        lo, hi = boot(cl, rng)
        nlo, nhi = naive(k, n)
        print(f"{label:<12}{n:>7,}{len(cl):>8,}{100*k/n:>8.1f}%"
              f"   {lo:5.1f} – {hi:5.1f}  (±{(hi-lo)/2:.1f}%p)"
              f"      {nlo:5.1f} – {nhi:5.1f}  (±{(nhi-nlo)/2:.1f}%p)")

    print(f"{NL}※ 스터디 클러스터는 개발셋 정답에 study 열이 없어 계산할 수 없다."
          f"{NL}   감사(AUDIT_2026-08-15 §3)에서 스터디 클러스터 구간은 펩타이드 클러스터의"
          f"{NL}   약 1.7배였으므로, 위 구간은 보수적이지 않은 쪽이다. 논문에 명시할 것.")


if __name__ == "__main__":
    main()
