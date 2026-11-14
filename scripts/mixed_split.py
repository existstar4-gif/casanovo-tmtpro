# -*- coding: utf-8 -*-
"""Phase B — peptide-wise split of TMT + naked MGFs into a mixed train/val/test set.

Leakage-0 rule (same as split_mgf_one.py): split = md5(bare peptide, mods stripped,
'-' removed, upper, I->L) % 100 -> <80 train, <90 val, else test. Same peptide always
same split, across BOTH TMT and naked (so a peptide seen labeled+naked can't leak).

Inputs : data/mgf_tmt\\*.mgf  +  data/mgf_naked\\*.mgf
Output : data/train_mixed\\{train,val,test}\\<stem>.mgf  (+ .done markers)
Then   : concat val/*.mgf -> train_mixed\\val_all.mgf (single file; casanovo val must be 1 file).
A 'cls' tag is added to TITLE-free tracking via filename only; SEQ already encodes TMT/naked.
"""
import os, glob, hashlib, re, time
from concurrent.futures import ProcessPoolExecutor, as_completed
WORKERS = 4   # was 8. Halved 2026-08-06: G: is a USB HDD that has been throwing
              # sporadic silent write corruption (one NUL-corrupted mzML in ~330
              # written on a "healthy" day, plus three during the 08-04 fault).
              # Each worker here holds one read plus THREE write streams
              # (train/val/test), so 8 workers meant 32 interleaved streams on one
              # head. Measured on build_mgf, the drive still had headroom at 2
              # workers (64% idle, queue 0.08), so 4 is a safe middle.

TMT_DIR   = r"data/mgf_tmt"
NAKED_DIR = r"data/mgf_naked"
OUTDIR    = r"data/train_mixed"
LOG       = r"out\mixed_split.log"
BRACKET = re.compile(r"\[[^\]]*\]")

def log(m):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def split_of(seq):
    bare = BRACKET.sub("", seq).replace("-", "").upper().replace("I", "L")
    h = int(hashlib.md5(bare.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")

def split_one(mgf, tag):
    stem = tag + "__" + os.path.splitext(os.path.basename(mgf))[0]
    done = os.path.join(OUTDIR, ".done", stem)
    if os.path.exists(done):
        return None
    tmp = {s: open(os.path.join(OUTDIR, s, stem + ".mgf.part"), "w", encoding="utf-8", newline="\n")
           for s in ("train", "val", "test")}
    counts = {s: 0 for s in tmp}
    block = []; seq = None
    with open(mgf, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("BEGIN IONS"):
                block = [line]; seq = None
            elif line.startswith("SEQ="):
                seq = line[4:].strip(); block.append(line)
            elif line.startswith("END IONS"):
                block.append(line)
                s = split_of(seq) if seq else "train"
                tmp[s].writelines(block); counts[s] += 1
                block = []
            else:
                block.append(line)
    for s, fh in tmp.items():
        fh.close()
        part = os.path.join(OUTDIR, s, stem + ".mgf.part")
        if counts[s] > 0:
            os.replace(part, os.path.join(OUTDIR, s, stem + ".mgf"))
        else:
            os.remove(part)
    open(done, "w").close()
    return counts

def _split_job(args):
    mgf, tag = args
    return tag, split_one(mgf, tag)

def main():
    for s in ("train", "val", "test", ".done"):
        os.makedirs(os.path.join(OUTDIR, s), exist_ok=True)
    jobs = [(p, "tmt") for p in sorted(glob.glob(os.path.join(TMT_DIR, "*.mgf")))] + \
           [(p, "naked") for p in sorted(glob.glob(os.path.join(NAKED_DIR, "*.mgf")))]
    log(f"=== mixed_split start: {len(jobs)} MGFs, {WORKERS} workers ===")
    tot = {"train": 0, "val": 0, "test": 0}
    by_tag = {"tmt": dict(train=0, val=0, test=0), "naked": dict(train=0, val=0, test=0)}
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_split_job, j) for j in jobs]
        for fut in as_completed(futs):
            tag, c = fut.result()
            done += 1
            if c:
                for s in tot: tot[s] += c[s]; by_tag[tag][s] += c[s]
            if done % 50 == 0 or done == len(jobs):
                log(f"[{done}/{len(jobs)}] tot={tot}")
    log(f"=== split done: {tot} | tmt={by_tag['tmt']} naked={by_tag['naked']} ===")

    # concat val into single file (casanovo val must be one file)
    val_all = os.path.join(OUTDIR, "val_all.mgf")
    log("concatenating val/*.mgf -> val_all.mgf ...")
    with open(val_all, "w", encoding="utf-8", newline="\n") as out:
        for vf in sorted(glob.glob(os.path.join(OUTDIR, "val", "*.mgf"))):
            with open(vf, encoding="utf-8") as fh:
                for line in fh:
                    out.write(line)
    log(f"val_all.mgf written -> {val_all}")

if __name__ == "__main__":
    main()
