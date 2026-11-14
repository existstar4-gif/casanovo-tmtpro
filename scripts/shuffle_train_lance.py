# -*- coding: utf-8 -*-
"""Rewrite train.lance with the rows interleaved, fixing the training-order defect.

WHY (diagnosed 2026-08-15):
  train.lance is stored in file order, and the file list is sorted, so all
  `naked__*` spectra (rows 0-13.7%) come first and all `tmt__*` spectra
  (13.7-100%) after -- and each study/run is a contiguous block within that.
  The training loader shuffles with a buffer of only 5,000 spectra (0.02% of
  21,036,675 rows), which cannot mix blocks that are millions of rows apart.
  So every epoch the model sees ~2.9M consecutive naked spectra, then ~18.1M
  consecutive TMT spectra, and every checkpoint is taken right after 18M
  TMT-only examples. That is textbook catastrophic forgetting, and it is the
  most parsimonious explanation for "fine-tuning damaged naked ability"
  (ours 43 vs stock Casanovo 60.5) -- the premise the gate/routing
  contributions rest on.

HOW:
  Sequential I/O only -- a random `take()` over 46 GB would thrash the disk.
  Open K sequential readers at evenly spaced offsets, pull a batch from a
  randomly chosen reader at a time, accumulate CHUNK batches, then permute
  rows across that whole chunk before writing. Rows that were millions apart
  end up adjacent, while every read stays sequential within its shard.

Result: naked/TMT and every study are spread uniformly, so the 5,000-spectrum
training buffer now samples a representative mix at every step.

Usage: python shuffle_train_lance.py <src.lance> <dst.lance> [K] [BATCH] [CHUNK]
"""
import sys, random, time

import lance
import pyarrow as pa

SRC = sys.argv[1] if len(sys.argv) > 1 else r"data/lance/train.lance"
DST = sys.argv[2] if len(sys.argv) > 2 else r"data/lance_interleaved/train.lance"
K = int(sys.argv[3]) if len(sys.argv) > 3 else 256      # sequential readers
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 256  # rows pulled per turn
CHUNK = int(sys.argv[5]) if len(sys.argv) > 5 else 64   # batches permuted together

SEED = 20260815
random.seed(SEED)

src = lance.dataset(SRC)
n = src.count_rows()
schema = src.schema
print(f"src={SRC} rows={n:,} K={K} batch={BATCH} chunk={CHUNK} seed={SEED}", flush=True)

bounds = []
for i in range(K):
    lo = i * n // K
    hi = (i + 1) * n // K
    if hi > lo:
        bounds.append((lo, hi - lo))

iters = [
    src.scanner(limit=length, offset=offset, batch_size=BATCH).to_batches()
    for offset, length in bounds
]

t0 = time.time()
state = {"rows": 0, "chunks": 0}


def gen():
    active = list(range(len(iters)))
    buf = []
    while active:
        i = random.choice(active)
        try:
            buf.append(next(iters[i]))
        except StopIteration:
            active.remove(i)
            continue
        if len(buf) >= CHUNK:
            yield _flush(buf)
            buf = []
    if buf:
        yield _flush(buf)


def _flush(buf):
    """Concatenate the collected batches and permute rows across all of them."""
    tbl = pa.Table.from_batches(buf)
    idx = list(range(tbl.num_rows))
    random.shuffle(idx)
    tbl = tbl.take(pa.array(idx))
    state["rows"] += tbl.num_rows
    state["chunks"] += 1
    if state["chunks"] % 50 == 0:
        el = time.time() - t0
        pct = 100.0 * state["rows"] / n
        eta = (el / max(pct, 1e-9)) * (100 - pct) / 60
        print(f"  {state['rows']:,} / {n:,} ({pct:.1f}%)  {el/60:.1f} min elapsed, ETA {eta:.0f} min", flush=True)
    return tbl.combine_chunks().to_batches(max_chunksize=BATCH * CHUNK)[0]


reader = pa.RecordBatchReader.from_batches(schema, gen())
lance.write_dataset(reader, DST, mode="create", max_rows_per_file=1_000_000)

out = lance.dataset(DST)
print(f"DONE rows={out.count_rows():,} (src {n:,})  {(time.time()-t0)/60:.1f} min", flush=True)
assert out.count_rows() == n, "ROW COUNT MISMATCH -- do not use this dataset"
print("row count verified OK", flush=True)
