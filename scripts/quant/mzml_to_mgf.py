# -*- coding: utf-8 -*-
"""mzml_to_mgf.py — 정량 경로용 MGF 변환. mzML 의 **MS2 전부**를 내보낸다.

기존 `mzml_subset_mgf.py` 는 정답지가 있는 스캔만 골라 평가셋을 만든다. 정량에서는 그것이
쓸 수 없다 — 이 파이프라인의 존재 이유가 **검색 엔진이 못 붙인 스펙트럼까지 정량하는 것**
이므로, 정답이 있는 것만 남기면 재려던 대상을 미리 지우는 셈이다.

TITLE 은 저장소 공통 규약 `<stem>:scan:<scan>` 으로 쓴다. 이 형식이라야 mzTab 의
spectra_ref index 를 (stem, scan) 으로 되돌릴 수 있고, `extract_reporters.py` 의
정량표와 같은 키로 붙는다(`combine_psm_quant.py`).

사용:
  python scripts/quant/mzml_to_mgf.py in.mzML -o out/quant/run.mgf
  python scripts/quant/mzml_to_mgf.py in.mzML -o run.mgf --limit 5000   # 시험용
"""
from __future__ import annotations

import argparse
import os
import re
import sys


def ms2_iter(path):
    """(scan, prec_mz, charge, mzs, ints). mzml_subset_mgf.py 와 같은 파싱 규칙."""
    from pyteomics import mzml
    with mzml.read(path) as rd:
        for s in rd:
            if s.get("ms level") != 2:
                continue
            m = re.search(r"scan=(\d+)", s.get("id", ""))
            if not m:
                continue
            pre = s.get("precursorList", {}).get("precursor", [{}])[0]
            si = pre.get("selectedIonList", {}).get("selectedIon", [{}])[0]
            mz = si.get("selected ion m/z")
            if mz is None:
                continue
            z = si.get("charge state")
            yield (int(m.group(1)), float(mz), int(z) if z else 0,
                   s["m/z array"], s["intensity array"])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("mzml")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만(시험용)")
    ap.add_argument("--min-peaks", type=int, default=0,
                    help="피크가 이보다 적은 MS2 는 건너뛴다. 기본 0 = 거르지 않는다")
    a = ap.parse_args()

    stem = os.path.basename(a.mzml)
    for suf in (".mzML", ".mzml"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]

    n = skipped = 0
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        for scan, mz, z, mzs, ints in ms2_iter(a.mzml):
            if a.min_peaks and len(mzs) < a.min_peaks:
                skipped += 1
                continue
            fh.write("BEGIN IONS\n")
            fh.write("TITLE=%s:scan:%d\n" % (stem, scan))
            fh.write("PEPMASS=%s\n" % mz)
            if z:
                fh.write("CHARGE=%d+\n" % z)
            fh.write("SCANS=%d\n" % scan)
            for x, y in zip(mzs, ints):
                fh.write("%.5f %.1f\n" % (x, y))
            fh.write("END IONS\n")
            n += 1
            if a.limit and n >= a.limit:
                break
    tail = "  (피크 부족 %d개 제외)" % skipped if skipped else ""
    print("%s: MS2 %s개 기록%s" % (a.out, format(n, ","), tail))


if __name__ == "__main__":
    main()
