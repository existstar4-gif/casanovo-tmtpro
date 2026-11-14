"""
build_mgf_one.py — 단일 mzML + 단일 stem 라벨 -> annotated MGF 파트 1개

build_mgf.py 의 검증된 로직을 1-raw 단위로 분리(병렬·재개·split-friendly).
  - 라벨 TSV 는 해당 raw_stem 만 담긴 작은 파일(out/labels_by_stem/<stem>.tsv).
  - 전구체 m/z 는 PD 보고값(prec_mz) 우선 — verify_match 수정#3 과 동일.
  - 출력은 임시파일(.part)에 쓰고 끝나면 원자적 rename → 중단 시 깨진 파일 방지(재개 안전).

usage:
  python scripts/build_mgf_one.py --mzml data/mzml/<stem>.mzML \
      --labels out/labels_by_stem/<stem>.tsv -o data/train/mgf/<stem>.mgf
exit: 0 정상, 3 라벨파일/ mzML 없음
"""
from __future__ import annotations
import argparse, os, sys
from pyteomics import mzml


def load_labels(path):
    labels = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        pm_i = col.get("prec_mz")
        for line in fh:
            c = line.rstrip("\n").split("\t")
            scan = c[col["scan"]]
            prec_mz = c[pm_i] if (pm_i is not None and pm_i < len(c)) else ""
            labels[scan] = (c[col["charge"]], c[col["proforma"]], prec_mz)
    return labels


def scan_of(spec):
    for part in spec.get("id", "").split():
        if part.startswith("scan="):
            return part.split("=", 1)[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mzml", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--min-peaks", type=int, default=20)
    args = ap.parse_args()

    if not os.path.exists(args.mzml) or not os.path.exists(args.labels):
        print(f"MISSING mzml/labels: {args.mzml}", file=sys.stderr)
        return 3
    labels = load_labels(args.labels)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".part"

    n_written = n_nolabel = n_fewpeaks = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as out:
        with mzml.read(args.mzml) as reader:
            for spec in reader:
                if spec.get("ms level") != 2:
                    continue
                scan = scan_of(spec)
                if scan is None or scan not in labels:
                    n_nolabel += 1
                    continue
                charge, seq, prec_mz = labels[scan]
                mzs = spec["m/z array"]; ints = spec["intensity array"]
                if len(mzs) < args.min_peaks:
                    n_fewpeaks += 1
                    continue
                prec = None
                if prec_mz:
                    try: prec = float(prec_mz)
                    except ValueError: prec = None
                if prec is None:
                    try:
                        prec = (spec["precursorList"]["precursor"][0]
                                ["selectedIonList"]["selectedIon"][0]["selected ion m/z"])
                    except (KeyError, IndexError):
                        prec = 0.0
                rt = spec.get("scanList", {}).get("scan", [{}])[0].get("scan start time")
                rt_sec = float(rt) * 60.0 if rt is not None else None
                stem = os.path.splitext(os.path.basename(args.mzml))[0]
                out.write("BEGIN IONS\n")
                out.write(f"TITLE={stem}:scan:{scan}\n")
                out.write(f"PEPMASS={prec}\n")
                out.write(f"CHARGE={charge}+\n")
                if rt_sec is not None:
                    out.write(f"RTINSECONDS={rt_sec}\n")
                out.write(f"SEQ={seq}\n")
                for mz, it in zip(mzs, ints):
                    out.write(f"{mz:.5f} {it:.2f}\n")
                out.write("END IONS\n")
                n_written += 1
    os.replace(tmp, args.out)
    print(f"{os.path.basename(args.out)}: wrote={n_written} nolabel_ms2={n_nolabel} fewpeaks={n_fewpeaks}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
