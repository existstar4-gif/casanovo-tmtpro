# -*- coding: utf-8 -*-
"""Build TMT annotated MGFs from regenerated mzML (G:) + TMT per-stem labels (C:).

F:-free regeneration, PARALLEL (process pool) since 417 stems sequentially is too slow.
Each worker builds one stem -> its own MGF (atomic .part rename); main process logs the
returned counts (no shared-log corruption). Skip-resume via existing output.
  mzML dir : data/mzml_tmt\\<stem>.mzML
  labels   : out\\labels_by_stem\\<stem>.tsv   (TMT labels, C:)
  out dir  : data/mgf_tmt\\<stem>.mgf
"""
import os, sys, glob, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MZDIR  = r"data/mzml_tmt"
# Read BOTH the original 417 stems and the NEW 322-stem expansion (F: labels).
LBLDIRS = [r"out\labels_by_stem",
           r"out\labels_by_stem_new",
           r"out\labels_by_stem_new2",
           r"out\labels_by_stem_k",
           r"out\labels_by_stem_phos"]
OUTDIR = r"data/mgf_tmt"
LOG    = r"out\build_mgf_tmt.log"
MIN_PEAKS = 20
WORKERS = 2   # was 8. Tuned 2026-08-05 by MEASUREMENT, not by analogy.
              # G: is a USB HDD, so the initial guess was that parallel reads would
              # thrash the head (as parallel raw conversion did earlier that day).
              # Measuring 1 worker disproved it: G: sat 82% IDLE, disk queue 0, and
              # only 23 MB/s read / 4.6 MB/s write -- a fraction of the drive. The
              # real bottleneck is single-threaded Python formatting of every peak.
              # Unlike ThermoRawFileParser (25 GB/process), pyteomics streams with
              # read(use_index=False), so extra workers cost little memory.

def build_stem(args):
    """Worker: build one stem's MGF. Returns (stem, status, written).

    Never lets an exception escape. A corrupt mzML raises lxml.etree.XMLSyntaxError,
    whose error-log object cannot be pickled back to the parent -- on 2026-08-06 that
    turned one bad file into a TypeError that killed the whole pool mid-run. Failures
    are reported as a plain string status so the run continues and names the bad file.
    """
    try:
        return _build_stem(args)
    except BaseException as e:
        return (args[0], "error", f"{type(e).__name__}: {e}"[:300])


def _build_stem(args):
    stem, lf = args
    from pyteomics import mzml
    import build_mgf_one as B
    mz = os.path.join(MZDIR, stem + ".mzML")
    out = os.path.join(OUTDIR, stem + ".mgf")
    if os.path.exists(out) and os.path.getsize(out) > 100:
        return (stem, "skip", 0)
    if not os.path.exists(mz):
        return (stem, "no_mzml", 0)
    labels = B.load_labels(lf)
    tmp = out + ".part"
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as o:
        with mzml.read(mz) as reader:
            for spec in reader:
                if spec.get("ms level") != 2:
                    continue
                scan = B.scan_of(spec)
                if scan is None or scan not in labels:
                    continue
                charge, seq, prec_mz = labels[scan]
                mzs = spec["m/z array"]; ints = spec["intensity array"]
                if len(mzs) < MIN_PEAKS:
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
                o.write("BEGIN IONS\n")
                o.write(f"TITLE={stem}:scan:{scan}\n")
                o.write(f"PEPMASS={prec}\n")
                o.write(f"CHARGE={charge}+\n")
                if rt_sec is not None:
                    o.write(f"RTINSECONDS={rt_sec}\n")
                o.write(f"SEQ={seq}\n")
                for mz_, it in zip(mzs, ints):
                    o.write(f"{mz_:.5f} {it:.2f}\n")
                o.write("END IONS\n")
                n += 1
    os.replace(tmp, out)
    return (stem, "ok", n)

def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = {}   # stem -> label path (first dir wins; dirs are disjoint anyway)
    for d in LBLDIRS:
        for p in sorted(glob.glob(os.path.join(d, "*.tsv"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            jobs.setdefault(stem, p)
    jobs = sorted(jobs.items())
    log(f"=== build_mgf_tmt start: {len(jobs)} stems, {WORKERS} workers ===")
    tot_w = done = skip = nomz = 0
    errors = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(build_stem, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            stem, status, n = fut.result()
            done += 1
            if status == "ok":
                tot_w += n
            elif status == "skip":
                skip += 1
            elif status == "no_mzml":
                nomz += 1
                log(f"NO mzML: {stem}")
            elif status == "error":
                errors.append((stem, n))
                log(f"ERROR {stem}: {n}")
            if done % 25 == 0:
                log(f"[{done}/{len(jobs)}] total_written={tot_w} skip={skip} no_mzml={nomz} err={len(errors)}")
    if errors:
        log(f"--- {len(errors)} stem(s) failed to build ---")
        for stem, msg in errors:
            log(f"    FAILED {stem}: {msg}")
    log(f"=== done: total_written={tot_w} skip={skip} no_mzml={nomz} errors={len(errors)} stems={len(jobs)} ===")

if __name__ == "__main__":
    main()
