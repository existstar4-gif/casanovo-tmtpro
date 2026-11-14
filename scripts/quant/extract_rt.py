# -*- coding: utf-8 -*-
"""extract_rt.py — mzML 에서 MS2 스캔 번호 → 머무름 시간(분) 표를 뽑는다 (2026-11-02, '우리만' 검증의 RT 일치 분석용).

mzTab 의 retention_time 이 비어 있어(null) 만든다. XML 을 파싱하지 않고 텍스트를 훑어 빠르게 읽는다.

    python scripts/quant/extract_rt.py --study study3
출력: out/quant/<study>/<stem>.rt.tsv  (scan, rt_min)
"""
import argparse
import glob
import os
import re

SCAN = re.compile(rb'<spectrum [^>]*?id="[^"]*?scan=(\d+)"')
RT = re.compile(rb'<cvParam [^>]*name="scan start time"[^>]*>')          # 속성 순서가 파일마다 달라 요소째 잡는다
VAL = re.compile(rb'value="([\d.eE+-]+)"')
UNIT = re.compile(rb'unitAccession="UO:(\d+)"')
MSL = re.compile(rb'name="ms level" value="(\d)"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="study3")
    ap.add_argument("--mzml-dir", default="data/mzml_tmt")
    a = ap.parse_args()
    stems = sorted({os.path.basename(p)[:-len("_psm.tsv")] for p in glob.glob(f"out/quant/{a.study}/*_psm.tsv")})
    for st in stems:
        out = f"out/quant/{a.study}/{st}.rt.tsv"
        if os.path.exists(out):
            continue
        rows = []
        buf = b""
        with open(f"{a.mzml_dir}/{st}.mzML", "rb") as fh:
            while True:
                chunk = fh.read(1 << 24)
                if not chunk:
                    break
                buf += chunk
                parts = buf.split(b"</spectrum>")
                buf = parts.pop()
                for p in parts:
                    m = SCAN.search(p)
                    if not m:
                        continue
                    lv = MSL.search(p)
                    if lv and lv.group(1) != b"2":
                        continue
                    r = RT.search(p)
                    if not r:
                        continue
                    el = r.group(0)
                    t = float(VAL.search(el).group(1))
                    u = UNIT.search(el)
                    if u and u.group(1) == b"0000010":    # 초 단위면 분으로
                        t /= 60.0
                    rows.append((int(m.group(1)), t))
        with open(out + ".tmp", "w", encoding="utf-8") as fh:
            fh.write("scan\trt_min\n")
            fh.writelines(f"{s}\t{t:.4f}\n" for s, t in rows)
        os.replace(out + ".tmp", out)
        print(st, len(rows), flush=True)


if __name__ == "__main__":
    main()
