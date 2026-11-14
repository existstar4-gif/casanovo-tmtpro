# -*- coding: utf-8 -*-
"""rescue_unified.py — 세 구제 실험을 하나의 정의로 채점한다.

이 스크립트가 존재하는 이유는, 같은 "구제율"이 실험마다 다른 정의로 계산되어
비교 불가능한 값끼리 나란히 보고됐기 때문이다(2026-08-28 발견). 고정한 규칙:

  1. GT/예측 키는 (파일 stem, scan). scan 만으로 잡으면 파일 간 scan 번호가 겹쳐
     라벨이 덮인다 — 실제로 PXD054085 에서 4,000 중 1,226 건이 덮여 있었다.
  2. I 와 L 은 target 판정과 학습 관측 판정 **양쪽 모두**에서 동일시한다.
     한쪽만 I=L 로 두면 '암기 비율'이 규칙 차이만으로 28%p 움직인다.
  3. 분모는 **시퀀싱된 예측 건수**로 통일한다(표본 스펙트럼 수가 아니라).
     두 값을 모두 출력하니 논문에서 어느 쪽을 쓸지는 한 번만 정하면 된다.
  4. 자체 데이터는 run 별 종(human/mouse)에 맞는 FASTA 로만 채점한다.
     두 종 합집합으로 채점하면 외부(단일 종)보다 표적 공간이 넓어 유리해진다.
  5. 순 구제율 = target 일치율 − decoy 일치율 (우연 일치 보정).
  6. ★ 학습 미관측 회수율에도 **같은 decoy 보정을 적용**한다(2026-08-29).
     이전에는 순 구제율만 decoy 를 빼고 미관측은 원시 계수를 써서 분자·분모의
     보정이 어긋나 있었다. 비례 귀속을 주값, 전량 귀속을 보수적 하한으로 병기한다.
  7. ★ 트립신 공간의 정의는 `build_tryptic_index.py` **하나뿐**이다(2026-09-15).
     이 파일이 자체 digest() 를 들고 있던 동안, 2026-09-11 에 추가한 시작 M 제거 규칙이
     재순위 인덱스(v2)에만 들어가고 구제 채점에는 빠져 한 논문 안에 "프로테옴에 존재한다"
     의 기준이 둘이 됐다. 이제 같은 함수를 import 하고, 실행마다 해시 집합이
     `out/tryptic_<종>_v2.npy` 와 일치하는지 대조해 찍는다.
     (missed cleavage 2 · 길이 6-50 은 원래부터 양쪽이 같았다 — 차이는 시작 M 뿐이었다.)

사용:
  python scripts/rescue_unified.py
"""
from __future__ import annotations

import os
import pickle
import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ★ 2026-09-15 — 디제스트를 인덱스 빌더에서 **직접 가져온다**.
# 예전에는 이 파일이 자체 digest() 를 들고 있었고, 2026-09-11 에 시작 M 제거 규칙을
# build_tryptic_index.py 에만 넣는 바람에 재순위(v2)와 구제(v1)가 서로 다른 트립신
# 공간을 쓰게 됐다. 정의를 한 곳으로 모아 그 종류의 어긋남을 구조적으로 막는다.
# (missed cleavage 2, 길이 6-50 은 원래부터 양쪽이 같았다 — 차이는 시작 M 뿐이었다.)
import build_tryptic_index as bti  # noqa: E402
# 참조 프로테옴 FASTA 를 둔 로컬 디렉터리. 환경변수로 덮어쓸 수 있다.
# 기본값은 다른 스크립트(rerank_grid, make_rerank_fig, dbrank_sweep_score)와 같은 경로로
# 맞춘다 — 여기만 "data/database" 라 2026-09-15 실행에서 FASTA 를 못 찾았다.
DB = os.environ.get("MASS_DB_DIR", "data/database")
FASTA = {
    "human": [f"{DB}/uniprotkb_AND_model_organism_9606_AND_r_2025_01_06.fasta"],
    "mouse": [f"{DB}/uniprotkb_mouse_AND_model_organism_1009_2025_01_23.fasta"],
}
# 절단 규칙은 build_tryptic_index.CUT 하나뿐이다(여기서 다시 정의하지 않는다).
NL = chr(10)
MOD = re.compile(r"\[[^\]]*\]")


def bare(s):
    return MOD.sub("", s).replace("-", "").strip()


def norm(s):
    """I/L 동일시 — target 판정과 학습 관측 판정에 **똑같이** 적용한다.

    정규화도 인덱스 빌더 것을 그대로 쓴다(대문자 + I→L). 둘이 갈라지면 판정이
    규칙 차이만으로 흔들린다.
    """
    return bti.norm(s)


def read_fasta(paths):
    seqs, buf = [], []
    for p in paths:
        for line in open(p, encoding="utf-8", errors="ignore"):
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                buf = []
            else:
                buf.append(line.strip())
        if buf:
            seqs.append("".join(buf))
            buf = []
    return seqs


def digest(seq):
    """트립신 디제스트 — 정의는 build_tryptic_index.digest 하나뿐이다.

    missed cleavage 2, 길이 6-50, 시작 M 제거 포함. Methods 에 적은 규칙과 같고
    재순위가 쓰는 `out/tryptic_*_v2.npy` 와 같은 공간을 만든다.
    """
    return set(bti.digest(seq, met_loss=True))


_IDX = {}
# (실험, 층, n, target, decoy, 학습미관측) — out/rescue_counts.tsv 로 나간다.
COUNTS = []


def index_for(species):
    if species in _IDX:
        return _IDX[species]
    t, d = set(), set()
    for s in read_fasta(FASTA[species]):
        for pep in digest(s):
            t.add(norm(pep))
        # decoy 는 뒤집은 단백질에 **똑같은 규칙**을 적용한다(검색 엔진의 reversed decoy 와
        # 같은 처리). 뒤집힌 서열은 M 으로 시작하는 일이 드물어 시작 M 제거의 이득을
        # 거의 받지 않는데, 그것이 바로 이 규칙이 우연 일치가 아니라는 근거다.
        for pep in digest(s[::-1]):
            d.add(norm(pep))
    d -= t
    _IDX[species] = (t, d)
    print(f"    [{species}] target {len(t):,} / decoy {len(d):,}", flush=True)
    _check_against_index(species, t)
    return _IDX[species]


def _check_against_index(species, t):
    """재순위가 쓰는 `out/tryptic_<종>_v2.npy` 와 같은 공간인지 확인한다.

    두 분석이 갈라졌던 것이 이번 수정의 발단이므로, 말로 맞췄다고 하지 말고
    실제 해시 집합이 일치하는지 매 실행마다 찍는다.
    """
    p = os.path.join(BASE, "out", f"tryptic_{species}_v2.npy")
    if not os.path.exists(p):
        print(f"    [{species}] ⚠️ {os.path.basename(p)} 없음 — 대조 생략", flush=True)
        return
    import numpy as np
    ref = np.load(p)
    here = np.fromiter({bti.h64(k) for k in t}, dtype=np.uint64)
    here.sort()
    same = len(here) == len(ref) and bool((here == ref).all())
    print(f"    [{species}] 재순위 인덱스 대조 {'일치' if same else '★불일치★'}"
          f" (여기 {len(here):,} / 인덱스 {len(ref):,})", flush=True)


def load(mztab, mgf):
    """(stem, scan, proforma, score, ppm) — MGF 순서로 index 를 되돌린다."""
    keys = []
    for line in open(mgf, encoding="utf-8"):
        if line.startswith("TITLE="):
            st, sc = line.strip()[6:].rsplit(":scan:", 1)
            keys.append((st, int(sc)))
    rows, hdr = [], None
    for line in open(mztab, encoding="utf-8", errors="replace"):
        if line.startswith("PSH"):
            hdr = line.rstrip("\n").split("\t")
            continue
        if not line.startswith("PSM") or not hdr:
            continue
        c = dict(zip(hdr, line.rstrip("\n").split("\t")))
        m = re.search(r"index=(\d+)", c.get("spectra_ref", ""))
        if not m:
            continue
        i = int(m.group(1))
        if i >= len(keys):
            continue

        def num(k):
            try:
                return float(c.get(k, 0))
            except (TypeError, ValueError):
                return 0.0

        exp, calc = num("exp_mass_to_charge"), num("calc_mass_to_charge")
        ppm = abs(exp - calc) / calc * 1e6 if calc else 9e9
        pro = (c.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence")
               or c.get("sequence", ""))
        rows.append((keys[i][0], keys[i][1], pro, num("search_engine_score[1]"), ppm))
    return rows, len(keys)


def score(rows, species_of, seen):
    """각 예측에 target/decoy/none 과 '학습에서 봤는가' 를 붙인다."""
    out = []
    for stem, scan, pro, sc, ppm in rows:
        sp = species_of(stem)
        if sp is None:
            continue
        t, d = index_for(sp)
        k = norm(bare(pro))
        cls = "target" if k in t else ("decoy" if k in d else "none")
        out.append({"stem": stem, "scan": scan, "pro": pro, "score": sc, "ppm": ppm,
                    "cls": cls, "seen": (k in seen) if seen else None})
    return out


def wilson(k, n, z=1.96):
    """이항 비율의 Wilson 95% 구간 (%)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (100 * (c - h), 100 * (c + h))


def summarise(name, recs, n_sampled):
    """★ 2026-08-29 정정 — decoy 보정의 비대칭을 없앴다.

    이전 판본은 순 구제율에는 decoy 를 빼면서(`(t-d)/n`) 학습 미관측 회수는
    target 원시 계수(`novel_n/n`)로 보고했다. decoy 율이 실험 간 0.8/1.4/0.6% 로
    2.3배 차이나므로 이는 공통 오프셋이 아니라 **차등 편향**이고, 그 결과 보고된
    "폭 0.3%p" 자체가 비대칭의 산물이었다. 이제 두 가지를 함께 낸다:

      비례 귀속 (주값)  novel × (t-d)/t
          decoy 일치가 관측/미관측 버킷에 target 과 같은 비율로 섞였다고 본다.
      전량 귀속 (하한)  (novel - d) / n
          decoy 일치가 전부 미관측 버킷에 떨어졌다고 보는 최악 가정.
          "decoy 는 정의상 미관측" 은 가정이지 정의가 아니므로 하한으로만 쓴다.
    """
    def block(label, sub):
        if not sub:
            return
        n = len(sub)
        t = sum(1 for r in sub if r["cls"] == "target")
        d = sum(1 for r in sub if r["cls"] == "decoy")
        net = 100 * (t - d) / n
        seen_t = [r for r in sub if r["cls"] == "target" and r["seen"] is not None]
        memo = 100 * sum(r["seen"] for r in seen_t) / len(seen_t) if seen_t else float("nan")
        novel_n = sum(1 for r in seen_t if not r["seen"])
        raw = 100 * novel_n / n
        prop = raw * (t - d) / t if t else float("nan")      # 비례 귀속
        full = 100 * max(0, novel_n - d) / n                 # 전량 귀속 (하한)
        lo, hi = wilson(novel_n, n)
        # 원시 계수를 남긴다 — 동질성 검정(카이제곱)을 본문 숫자에서 역산하지 않고
        # 계수에서 직접 하기 위해서다(2026-09-15).
        COUNTS.append((name, label.strip(), n, t, d, novel_n))
        print(f"    {label:<26} n={n:>6,}  target {100*t/n:>5.1f}%  decoy {100*d/n:>4.1f}%"
              f"  순 {net:>5.1f}%  암기 {memo:>5.1f}%")
        print(f"    {'':<26} 미관측  원시 {novel_n:>5,} ({raw:4.1f}%)  "
              f"비례 {prop:4.2f}%  전량 {full:4.2f}%  95%CI(원시) {lo:.2f}-{hi:.2f}%")

    print(f"{NL}  {name}   시퀀싱 예측 {len(recs):,} / 표본 스펙트럼 {n_sampled:,}")
    block("전체 예측", recs)
    mass = [r for r in recs if r["ppm"] <= 20.0]
    block("전구체 질량 일치 ≤20ppm", mass)
    mass.sort(key=lambda r: -r["score"])
    block("  ↳ 점수 상위 50%", mass[: len(mass) // 2])
    block("  ↳ 점수 상위 10%", mass[: len(mass) // 10])
    uniq = {norm(bare(r["pro"])) for r in recs if r["cls"] == "target"}
    print(f"    회수 고유 펩타이드 {len(uniq):,}")


def main():
    # ⚠️ 2026-08-28: _train_peptides.pkl 은 학습 분할이 아니었다 (15.0% 가 val/test).
    #    train_peptides_true.pkl = 라벨 코퍼스에 split_of 를 적용한 train 버킷
    #    = 실제 학습 펩타이드의 상위집합 → "학습에 없음" 은 보수적 하한.
    pkl = os.path.join(BASE, "out", "train_peptides_true.pkl")
    seen_raw = pickle.load(open(pkl, "rb")) if os.path.exists(pkl) else set()
    seen = {norm(p) for p in seen_raw}          # 규칙 2 — 학습 집합도 I=L
    print(f"학습 펩타이드(상위집합) {len(seen_raw):,} (I=L 정규화 후 {len(seen):,})")

    # 자체 6 run — GATE=off 재예측분. run 별 종에 맞는 FASTA 로만 채점한다.
    sp_map = {}
    for line in open(os.path.join(BASE, "out", "rescue_runs.txt"), encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 2:
            sp_map[c[1]] = c[0]
    print("\n[1] 자체 6 run vs Proteome Discoverer  (GATE=off, rescue_v2.mztab)")
    rows, n = load("data/eval/rescue_v2.mztab", "data/eval/unidentified_v1.mgf")
    summarise("자체 6 run", score(rows, lambda s: sp_map.get(s), seen), n)

    # 외부 — FragPipe 미동정분
    for acc, sp, keep in (("PXD054085", "mouse", None),
                          ("PXD063976", "human", "Patient1_Phos_F_search")):
        d = f"data/external/{acc}"
        if not os.path.exists(f"{d}/unid_fp.mztab"):
            continue
        print(f"\n[2] {acc} vs FragPipe")
        rows, n = load(f"{d}/unid_fp.mztab", f"{d}/unid_fp.mgf")
        if keep:
            before = len(rows)
            rows = [r for r in rows if r[0] == keep]
            print(f"    미검색 파일의 예측 {before-len(rows):,}건 제외")
        summarise(acc, score(rows, lambda s: sp, seen), n)

    out = os.path.join(BASE, "out", "rescue_counts.tsv")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("experiment\tstratum\tn\ttarget\tdecoy\tnovel\n")
        for row in COUNTS:
            fh.write("\t".join(str(v) for v in row) + "\n")
    print(f"{NL}원시 계수 {len(COUNTS)}행 -> {out}")


if __name__ == "__main__":
    main()
