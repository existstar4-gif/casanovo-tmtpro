# -*- coding: utf-8 -*-
"""build_tryptic_index.py — 참조 프로테옴 트립신 디제스트를 저메모리 인덱스로 굽는다.

왜: 디코딩 재순위에서 "이 후보가 트립신 공간에 있는가"를 스펙트럼마다 물어야 하는데,
펩타이드 문자열 집합을 그대로 들면 사람 프로테옴 기준 400~700 MB 가 된다. 이 환경은
RAM 경합으로 여러 번 죽은 이력이 있어(2026-09-03 기준 여유 4.9 GB) 그 비용을 감당하기 어렵다.

대신 각 펩타이드의 blake2b 64비트 다이제스트만 정렬해 uint64 배열로 저장한다.
5백만 펩타이드 = 40 MB. 조회는 np.searchsorted 로 O(log n).

⚠️ 해시 충돌: 64비트에 5e6 개면 생일 문제로 충돌 기대값 약 7e-7 건이다. 무시할 수 있다.
⚠️ 정규화는 반드시 mixed_split.py::split_of 와 같아야 한다(대문자 + I→L). 한쪽만 I=L 로
   두면 판정이 규칙 차이만으로 흔들린다 — rescue_unified.py 가 같은 이유로 못 박아 두었다.

디제스트 규칙은 Methods 와 같다: 트립신, missed cleavage 2, 길이 6-50.

사용:
  python scripts/build_tryptic_index.py --fasta A.fasta [B.fasta ...] -o out/tryptic_human.npy
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

MISSED_CLEAVAGES = 2
LEN_LO, LEN_HI = 6, 50
# 트립신: K 또는 R 뒤에서 자르되 P 가 뒤따르면 자르지 않는다
CUT = re.compile(r"(?<=[KR])(?!P)")


def norm(seq: str) -> str:
    """mixed_split.py::split_of 와 같은 정규화 — 대문자 + I→L."""
    return seq.upper().replace("I", "L")


def h64(seq: str) -> int:
    return int.from_bytes(hashlib.blake2b(seq.encode(), digest_size=8).digest(), "big")


def read_fasta(path):
    name, buf = None, []
    with io.open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield "".join(buf)
                name, buf = line, []
            else:
                buf.append(line.strip())
    if name is not None:
        yield "".join(buf)


def digest(protein: str, met_loss: bool = True):
    """트립신 절단 + missed cleavage. 길이 범위 밖은 버린다.

    met_loss: 단백질이 M 으로 시작하면 그 M 이 잘린 N말단 펩타이드도 넣는다. 검색 엔진이
    기본으로 고려하는 규칙이다. 빠져 있던 판(2026-09-11 이전)에서는 내부 test set 의 'DB 밖'
    정답 17건이 **전부** 이것이었고, PXD072211 의 125건 중 42건이 이것이었다.
    """
    parts = [p for p in CUT.split(protein) if p]
    for i in range(len(parts)):
        acc = []
        for j in range(i, min(i + MISSED_CLEAVAGES + 1, len(parts))):
            acc.append(parts[j])
            pep = "".join(acc)
            if LEN_LO <= len(pep) <= LEN_HI:
                yield pep
    if met_loss and protein.startswith("M") and parts:
        acc = []
        for j in range(0, min(MISSED_CLEAVAGES + 1, len(parts))):
            acc.append(parts[j])
            pep = "".join(acc)[1:]
            if LEN_LO <= len(pep) <= LEN_HI:
                yield pep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", nargs="+", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--no-met-loss", action="store_true",
                    help="시작 M 제거 펩타이드를 넣지 않는다(2026-09-11 이전 인덱스 재현용)")
    a = ap.parse_args()

    seen = set()
    n_prot = 0
    for f in a.fasta:
        if not os.path.exists(f):
            raise SystemExit("없음: " + f)
        for prot in read_fasta(f):
            n_prot += 1
            if not prot:
                continue
            for pep in digest(prot, met_loss=not a.no_met_loss):
                seen.add(h64(norm(pep)))
        print(f"  {os.path.basename(f)}  누적 단백질 {n_prot:,}  고유 펩타이드 {len(seen):,}")

    arr = np.fromiter(seen, dtype=np.uint64, count=len(seen))
    arr.sort()
    np.save(a.out, arr)
    print(f"\n{a.out}  {len(arr):,} 펩타이드  {arr.nbytes/1048576:.1f} MB")


if __name__ == "__main__":
    main()
