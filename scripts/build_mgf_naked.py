# -*- coding: utf-8 -*-
"""Build naked annotated MGFs: for each converted naked mzML + its per-stem labels.

Reuses the proven per-spectrum logic from build_mgf_one.py (same precursor m/z =
PD-reported prec_mz, min-peaks filter, atomic .part rename). Output SEQ has NO TMT.

  mzML dir   : data/mzml_naked\\<stem>.mzML
  labels dir : out\\labels_by_stem_naked\\<stem>.tsv
  out dir    : data/train\\mgf_naked\\<stem>.mgf
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyteomics import mzml
import build_mgf_one as B

MZDIR  = r"data/mzml_naked"
LBLDIRS = [r"out\labels_by_stem_naked",
           r"out\labels_by_stem_naked2"]
OUTDIR = r"data/mgf_naked"
LOG    = r"out\build_mgf_naked.log"
MIN_PEAKS = 20

os.makedirs(OUTDIR, exist_ok=True)

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def build_one(mzml_path, labels_path, out_path):
    """Mirror of build_mgf_one.main body (one stem), returns (written, nolabel, fewpeaks)."""
    labels = B.load_labels(labels_path)
    tmp = out_path + ".part"
    n_written = n_nolabel = n_fewpeaks = 0
    stem = os.path.splitext(os.path.basename(mzml_path))[0]
    with open(tmp, "w", encoding="utf-8", newline="\n") as out:
        with mzml.read(mzml_path) as reader:
            for spec in reader:
                if spec.get("ms level") != 2:
                    continue
                scan = B.scan_of(spec)
                if scan is None or scan not in labels:
                    n_nolabel += 1
                    continue
                charge, seq, prec_mz = labels[scan]
                mzs = spec["m/z array"]; ints = spec["intensity array"]
                if len(mzs) < MIN_PEAKS:
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
    os.replace(tmp, out_path)
    return n_written, n_nolabel, n_fewpeaks

def main():
    label_files = []
    for d in LBLDIRS:
        label_files += sorted(glob.glob(os.path.join(d, "*.tsv")))
    log(f"=== build_mgf_naked start: {len(label_files)} stems ({len(LBLDIRS)} label dirs) ===")
    tot_w = tot_skip = tot_missing = 0
    for i, lf in enumerate(label_files, 1):
        stem = os.path.splitext(os.path.basename(lf))[0]
        mz = os.path.join(MZDIR, stem + ".mzML")
        out = os.path.join(OUTDIR, stem + ".mgf")
        if os.path.exists(out) and os.path.getsize(out) > 100:
            log(f"[{i}/{len(label_files)}] SKIP (exists) {stem}")
            tot_skip += 1
            continue
        if not os.path.exists(mz):
            log(f"[{i}/{len(label_files)}] NO mzML (not converted) {stem}")
            tot_missing += 1
            continue
        t0 = time.time()
        w, nl, fp = build_one(mz, lf, out)
        log(f"[{i}/{len(label_files)}] OK {stem}: wrote={w} nolabel_ms2={nl} fewpeaks={fp} ({time.time()-t0:.0f}s)")
        tot_w += w
    log(f"=== done: total_written={tot_w} skipped={tot_skip} no_mzml={tot_missing} ===")

if __name__ == "__main__":
    main()
