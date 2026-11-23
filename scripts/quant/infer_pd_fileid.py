# -*- coding: utf-8 -*-
"""infer_pd_fileid.py — InputFiles.txt 없는 PD PSMs export 의 File ID → 분획 stem 을 데이터로 찾는다 (2026-11-17).

Total 3개 스터디의 PSMs export(I:)에는 InputFiles.txt 가 없다. File ID(F3.1 …)의 순서를 분획 번호로
가정하지 않고, File ID 마다 PSM 200개의 (First Scan, 이론 m/z)를 각 분획의 rep.tsv (scan, prec_mz)와
대조해 20 ppm 안에서 맞는 비율이 가장 높은 분획을 고른다. 1위가 90% 이상, 2위가 10% 미만일 때만 받는다.

    python scripts/quant/infer_pd_fileid.py --study study3t --pd "data/pd/study3_total_fr__PSMs.txt"
"""
from __future__ import annotations

import argparse
import collections
import glob
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from count_matched_fdr import pp      # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
BASE = os.path.dirname(os.path.dirname(HERE))
PROTON = 1.007276


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--pd", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    reps = {}
    for p in sorted(glob.glob(os.path.join(BASE, "out/quant", a.study, "*.rep.tsv"))):
        stem = os.path.basename(p)[:-len(".rep.tsv")]
        d = {}
        for line in io.open(p, encoding="utf-8"):
            if line.startswith("#") or line.startswith("stem\t"):
                continue
            c = line.split("\t", 5)
            d[int(c[1])] = float(c[3])
        reps[stem] = d

    mapping = {}
    for path in a.pd:
        by = collections.defaultdict(list)
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            hdr = [pp.clean_field(x) for x in fh.readline().rstrip("\n").split("\t")]
            ix = {x: i for i, x in enumerate(hdr)}
            for line in fh:
                c = [pp.clean_field(x) for x in line.rstrip("\n").split("\t")]
                z = int(c[ix["Charge"]]); mh = float(c[ix["Theo. MH+ [Da]"]])
                by[c[ix["File ID"]]].append((int(c[ix["First Scan"]]), (mh - PROTON + z * PROTON) / z))
        rnd = random.Random(454)
        for fid, rows in sorted(by.items()):
            smp = rnd.sample(rows, min(200, len(rows)))
            score = []
            for stem, d in reps.items():
                ok = sum(1 for sc, mz in smp if sc in d and abs(d[sc] - mz) / mz * 1e6 < 20)
                score.append((ok / len(smp), stem))
            score.sort(reverse=True)
            (s1, best), (s2, _) = score[0], score[1]
            good = s1 >= 0.9 and s2 < 0.1
            print(f"  {os.path.basename(path)[-14:]:>14} {fid:>6} -> {best[-6:]}  match {100 * s1:.0f}% (next {100 * s2:.0f}%) {'ok' if good else 'REJECT'}")
            if good:
                if fid in mapping and mapping[fid] != best:
                    sys.exit(f"conflict for {fid}: {mapping[fid]} vs {best}")
                mapping[fid] = best
    json.dump(mapping, open(a.out, "w"), indent=1)
    print(f"{len(mapping)} File IDs mapped -> {a.out}")


if __name__ == "__main__":
    main()
