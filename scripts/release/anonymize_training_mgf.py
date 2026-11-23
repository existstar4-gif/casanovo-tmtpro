# -*- coding: utf-8 -*-
"""학습 코퍼스(annotated MGF) 공개용 익명화 (2026-11-19).

원본 파일 이름과 TITLE 에는 날짜 · 세포주 · 유전형 · 처리 · 조직 · 실험 방식 · 반복 번호가 들어 있다
(예: 20250101_CELLLINE_GENEOE_treatment_Total_TMT_16ul_fr_19). 피크 · 전구체 · 전하 · 정답 서열(SEQ)은
그대로 두고 이름만 바꾼다. 원본은 읽기만 한다.

두 방식:
  rename (기본)  파일 이름과 TITLE 을 S{스터디}_F{파일} 로 바꾼다. 파일 구조 · 분할이 원본과 같아
                 재학습 재현성이 가장 좋다. 다만 같은 스터디의 파일은 같은 S 번호로 묶인다
                 → 스터디별 검출 펩타이드 목록은 다시 만들 수 있다.
  pool           분할(train/val/test)마다 모든 스펙트럼을 섞어 크기가 같은 묶음 파일로 낸다.
                 TITLE 은 일련번호뿐이라 스터디 · 파일 구분이 사라진다. 대신 파일 단위 구조는 없어진다.

공통:
  - tmt__ / naked__ 접두어는 남긴다 (rename). 표지 종류일 뿐 스터디 정보가 아니고, 학습 스크립트가 쓴다.
  - 스터디 번호는 고정 시드로 무작위 배정한다 (S001 이 가장 오래된 스터디가 아니게).
  - --drop-rt 면 RTINSECONDS 줄을 지운다 (모델 입력이 아니다).
  - 대응표(원래 이름 ↔ 새 이름)는 --map 으로 따로 저장한다. **이 파일은 공개하지 않는다.**
  - 끝나면 검증한다: 스펙트럼 수, 그리고 TITLE(·RT)을 뺀 내용의 해시가 원본과 같은지.
    rename 은 파일마다 순서까지, pool 은 분할마다 순서와 무관한 합 해시로 비교한다.

사용:
  python anonymize_training_mgf.py data\\train_mixed data\\train_mixed_anon ^
         --manifest out\\train_data_manifest.tsv --map private\\anon_map.tsv --dry-run
  (--dry-run 을 빼면 실제로 쓴다. --limit-files N 으로 앞의 N 개 파일만 시험할 수 있다.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import tempfile
from pathlib import Path

SPLITS = ("train", "val", "test")
KINDS = ("tmt__", "naked__")
SEED = 20261119


def split_kind(name: str):
    for k in KINDS:
        if name.startswith(k):
            return k, name[len(k):]
    return "", name


def study_of(stem: str, studies: list) -> str:
    """매니페스트의 스터디 이름 중 가장 긴 접두어. 없으면 분획 꼬리를 떼어 낸 이름."""
    best = ""
    for s in studies:
        if stem.startswith(s) and len(s) > len(best):
            best = s
    if best:
        return best
    return re.sub(r"(?i)[_-](fr|f|rep)?_?\d+$", "", stem)


def build_map(src: Path, manifest: Path | None):
    """{원래 파일 이름(접두어 포함, .mgf 제외): (스터디, 새 이름)}"""
    studies = []
    if manifest and manifest.is_file():
        with open(manifest, encoding="utf-8") as fh:
            studies = [r["study"] for r in csv.DictReader(fh, delimiter="\t") if r.get("study")]
    names = set()
    for sp in SPLITS:
        d = src / sp
        if d.is_dir():
            names.update(p.stem for p in d.glob("*.mgf"))
    by_study = {}
    for n in sorted(names):
        _, bare = split_kind(n)
        by_study.setdefault(study_of(bare, studies), []).append(n)
    order = sorted(by_study)
    random.Random(SEED).shuffle(order)
    m = {}
    for si, st in enumerate(order, 1):
        for fi, n in enumerate(sorted(by_study[st]), 1):
            kind, _ = split_kind(n)
            m[n] = (st, f"{kind}S{si:03d}_F{fi:03d}")
    return m, len(studies)


def spectra(path: Path):
    """BEGIN IONS … END IONS 블록을 줄 목록으로."""
    block = None
    with open(path, encoding="utf-8", errors="strict") as fh:
        for line in fh:
            if line.startswith("BEGIN IONS"):
                block = [line]
            elif block is not None:
                block.append(line)
                if line.startswith("END IONS"):
                    yield block
                    block = None


def content_hash(block, drop_rt) -> bytes:
    h = hashlib.sha256()
    for line in block:
        if line.startswith("TITLE=") or (drop_rt and line.startswith("RTINSECONDS=")):
            continue
        h.update(line.encode("utf-8"))
    return h.digest()


def rewrite(block, title, drop_rt):
    out = []
    for line in block:
        if line.startswith("TITLE="):
            out.append(f"TITLE={title}\n")
        elif drop_rt and line.startswith("RTINSECONDS="):
            continue
        else:
            out.append(line)
    return out


_TITLE = re.compile(r"^TITLE=(.*):scan:(\d+)\s*$")


def do_rename(src, dst, mapping, files, drop_rt):
    """파일 하나씩: 이름 · TITLE 을 바꾸고 내용 해시를 바로 비교한다."""
    total = 0
    for sp, p in files:
        new = mapping[p.stem][1]
        out = dst / sp / (new + ".mgf")
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(".mgf.part")
        h_in, h_out, n = hashlib.sha256(), hashlib.sha256(), 0
        _, bare = split_kind(p.stem)
        _, new_bare = split_kind(new)
        with open(part, "w", encoding="utf-8", newline="") as fo:
            for b in spectra(p):
                t = next((l for l in b if l.startswith("TITLE=")), "")
                m = _TITLE.match(t)
                if not m or m.group(1) != bare:
                    raise SystemExit(f"unexpected TITLE in {p}: {t.strip()!r}")
                h_in.update(content_hash(b, drop_rt))
                nb = rewrite(b, f"{new_bare}:scan:{m.group(2)}", drop_rt)
                fo.writelines(nb)
                n += 1
        for b in spectra(part):                       # 쓴 파일을 다시 읽어 확인
            h_out.update(content_hash(b, drop_rt))
        if h_in.digest() != h_out.digest():
            raise SystemExit(f"content mismatch after rewrite: {p}")
        os.replace(part, out)
        total += n
        print(f"  {sp}/{p.name} -> {sp}/{out.name}  {n:,} spectra  ok", flush=True)
    return total


def do_pool(src, dst, files, drop_rt, per_file):
    """분할마다: 1차로 스펙트럼을 무작위 통 K 개에 나눠 담고, 2차로 통 안을 섞어 일련번호 TITLE 로 쓴다."""
    rng = random.Random(SEED)
    total = 0
    for sp in SPLITS:
        sp_files = [p for s, p in files if s == sp]
        if not sp_files:
            continue
        n_all = 0
        tmp = Path(tempfile.mkdtemp(prefix=f"pool_{sp}_", dir=dst))
        # 통 수: 한 통을 메모리에서 섞을 수 있을 만큼 (통 하나 ≈ per_file 스펙트럼)
        est = sum(p.stat().st_size for p in sp_files) // 9000 + 1          # 스펙트럼당 약 9 kB
        k = max(1, est // per_file + 1)
        bins = [open(tmp / f"b{i:04d}", "w", encoding="utf-8", newline="") for i in range(k)]
        sum_in = 0
        for p in sp_files:
            for b in spectra(p):
                sum_in = (sum_in + int.from_bytes(content_hash(b, drop_rt), "big")) % (1 << 256)
                bins[rng.randrange(k)].writelines(b)
                n_all += 1
            print(f"  {sp}: read {p.name}", flush=True)
        for f in bins:
            f.close()
        (dst / sp).mkdir(parents=True, exist_ok=True)
        idx, sum_out = 0, 0
        for i in range(k):
            bpath = tmp / f"b{i:04d}"
            blocks = list(spectra(bpath))
            rng.shuffle(blocks)
            out = dst / sp / f"pool_{sp}_{i + 1:04d}.mgf"
            with open(out, "w", encoding="utf-8", newline="") as fo:
                for b in blocks:
                    idx += 1
                    nb = rewrite(b, f"pool_{sp}:{idx}", drop_rt)
                    sum_out = (sum_out + int.from_bytes(content_hash(nb, drop_rt), "big")) % (1 << 256)
                    fo.writelines(nb)
            bpath.unlink()
        tmp.rmdir()
        if idx != n_all or sum_in != sum_out:
            raise SystemExit(f"{sp}: verification failed ({n_all} in / {idx} out)")
        print(f"  {sp}: {n_all:,} spectra -> {k} pooled files  ok", flush=True)
        total += n_all
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="train/ val/ test/ 가 있는 원본 폴더 (읽기만 한다)")
    ap.add_argument("dst", type=Path, help="익명화본을 쓸 새 폴더")
    ap.add_argument("--manifest", type=Path, help="train_data_manifest.tsv (스터디 묶음용)")
    ap.add_argument("--map", type=Path, required=True, help="대응표 저장 위치 — 비공개로 보관")
    ap.add_argument("--mode", choices=("rename", "pool"), default="rename")
    ap.add_argument("--drop-rt", action="store_true", help="RTINSECONDS 줄을 지운다")
    ap.add_argument("--per-file", type=int, default=100_000, help="pool: 묶음 파일 하나의 스펙트럼 수")
    ap.add_argument("--limit-files", type=int, default=0, help="시험용: 분할마다 앞의 N 개 파일만")
    ap.add_argument("--dry-run", action="store_true", help="대응표와 계획만 보여 주고 쓰지 않는다")
    a = ap.parse_args()

    if a.dst.resolve() == a.src.resolve() or a.src.resolve() in a.dst.resolve().parents:
        raise SystemExit("dst must be a new folder outside src")
    mapping, n_manifest = build_map(a.src, a.manifest)
    studies = sorted({v[0] for v in mapping.values()})
    print(f"files: {len(mapping):,}  studies: {len(studies)}  (manifest studies: {n_manifest})")
    files = []
    for sp in SPLITS:
        lst = sorted((a.src / sp).glob("*.mgf")) if (a.src / sp).is_dir() else []
        if a.limit_files:
            lst = lst[:a.limit_files]
        files += [(sp, p) for p in lst]
    size = sum(p.stat().st_size for _, p in files)
    print(f"to process: {len(files):,} files, {size / 2**30:.1f} GB  mode={a.mode}  drop_rt={a.drop_rt}")
    skipped = [p.name for p in a.src.glob("*.mgf")]
    if skipped:
        print("not processed (files directly in src, e.g. merged copies):", ", ".join(skipped))
    for n, (st, new) in list(sorted(mapping.items(), key=lambda kv: kv[1][1]))[:8]:
        print(f"  {new:<22} <- {n}")
    if a.dry_run:
        return
    a.map.parent.mkdir(parents=True, exist_ok=True)
    with open(a.map, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["original_file", "study", "anonymized_file"])
        for n, (st, new) in sorted(mapping.items()):
            w.writerow([n, st, new if a.mode == "rename" else "(pooled)"])
    print(f"map written (keep private): {a.map}")
    a.dst.mkdir(parents=True, exist_ok=True)
    if a.mode == "rename":
        n = do_rename(a.src, a.dst, mapping, files, a.drop_rt)
    else:
        n = do_pool(a.src, a.dst, files, a.drop_rt, a.per_file)
    print(f"DONE {n:,} spectra, all verified")


if __name__ == "__main__":
    sys.exit(main())
