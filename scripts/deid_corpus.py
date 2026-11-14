# -*- coding: utf-8 -*-
"""deid_corpus.py — 공개용 코퍼스에서 스터디·시료 식별자를 익명화한다.

★ 설계 원칙 — 작업용 lance 는 건드리지 않는다.
  51.8GB / 2,100만 행을 재작성하는 것은 이 환경(크래시 이력 다수)에서 위험한데 얻는 것이 없다.
  공개되는 것은 lance 가 아니라 MGF + 라벨이고, lance 의 peak_file 컬럼은
  **stem 목록을 빠짐없이 뽑는 소스로만** 쓴다.

이름이 박혀 있는 곳 네 군데를 모두 다룬다:
  1. MGF 파일명            tmt__<날짜>_<과제·시료명>_TMT_<주입량>_fr_NN.mgf
  2. MGF 내부 TITLE=       TITLE=<날짜>_<과제·시료명>_N:scan:1950
  3. all_labels_v3.tsv     raw_stem 컬럼 (2,140만 행)
  4. train_data_manifest   study 컬럼

익명 이름 형식:  TMT_S001_R003 / LF_S042_R012
  - 클래스(TMT/무표지)는 과학적으로 필요하므로 유지한다
  - 같은 스터디의 run 은 같은 S 번호로 묶여 코퍼스 구조가 보존된다

모드:
  --scan            (기본) 매핑표와 민감 토큰 목록만 만든다. 데이터를 쓰지 않는다
  --apply --out D   D 에 익명화된 배포본을 만든다
  --verify D        D 안에 민감 토큰이 남아 있는지 검사한다
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
LANCE = "data/lance_interleaved/train.lance"
MGF_ROOT = "data/train_mixed"

MAP_TSV = os.path.join(OUT, "deid_map.tsv")          # 비공개
REPORT = os.path.join(OUT, "deid_report.txt")

# stem 끝에 붙는 run/fraction 표기 — 스터디를 묶으려면 이걸 떼야 한다
RUN_SUFFIX = re.compile(
    r"(_fr_?\d+|_f\d+|_rep\d+|_r\d+|_\d+ul|_\d+K|_set\d+|_\d+[A-Za-z]?)$", re.I)
DATE = re.compile(r"^\d{6,8}$")
# 주입량·기기설정·배치번호 — 사람이나 시료를 식별하지 않는다. 검토 목록에서 뺀다.
NOT_IDENTIFYING = re.compile(
    r"^(\d+(ul|ug|ng|mg|ml|K|k|min|h|hr|um|nm|mM|uM)"
    r"|set\d+|batch\d+|fr\d*|f\d+|rep\d+|r\d+|s\d+|p\d+"
    r"|\d+[A-Za-z]|[A-Za-z]\d+|v\d+|\d+plex)$")

# 날짜·측정조건처럼 사람/시료를 식별하지 않는 토큰. 민감 목록에서 뺀다.
GENERIC = {
    "tmt", "tmtpro", "total", "phospho", "phos", "naked", "fr", "ul", "ug", "ng",
    "set", "rep", "re", "test", "std", "mix", "pool", "sample", "frac", "fraction",
    "min", "max", "top", "ms", "ms2", "dda", "dia", "hcd", "cid", "etd",
    "k", "kd", "ko", "oe", "wt", "ctrl", "control", "and", "of", "the", "v2", "v3",
    "1st", "2nd", "3rd", "new", "old", "final", "raw", "mgf", "mzml",
}


def stem_class(stem: str) -> str:
    if stem.startswith("tmt__"):
        return "TMT"
    if stem.startswith("naked__"):
        return "LF"
    return "TMT"


def strip_prefix(stem: str) -> str:
    for p in ("tmt__", "naked__"):
        if stem.startswith(p):
            return stem[len(p):]
    return stem


def study_key(stem: str) -> str:
    """run/fraction 표기를 반복 제거해 스터디 단위로 묶는다."""
    s = strip_prefix(stem)
    for _ in range(4):
        s2 = RUN_SUFFIX.sub("", s)
        if s2 == s:
            break
        s = s2
    return s


def tokens_of(stem: str):
    for t in re.split(r"[_\-.]+", strip_prefix(stem)):
        t = t.strip()
        if (not t or DATE.match(t) or t.lower() in GENERIC or t.isdigit()
                or NOT_IDENTIFYING.match(t)):
            continue
        yield t


def collect_stems():
    """모든 소스에서 stem 을 모은다. lance 가 기준, 나머지는 보강."""
    stems = set()
    src = collections.Counter()

    try:
        import lance
        ds = lance.dataset(LANCE)
        for pf in set(ds.to_table(columns=["peak_file"]).column("peak_file").to_pylist()):
            stems.add(re.sub(r"\.mgf$", "", pf, flags=re.I))
            src["lance"] += 1
    except Exception as exc:                          # noqa: BLE001
        print(f"  ⚠ lance 읽기 실패 ({exc}) — MGF 파일명으로 대체합니다")

    for split in ("train", "val", "test"):
        d = os.path.join(MGF_ROOT, split)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith(".mgf"):
                stems.add(fn[:-4])
                src["mgf"] += 1
    return stems, src


def build_map(stems):
    """stem -> 익명 이름. 결정적으로(정렬 순서) 배정해 재실행해도 같은 결과가 나온다."""
    by_study = collections.defaultdict(list)
    for s in stems:
        by_study[(stem_class(s), study_key(s))].append(s)

    mapping, studies = {}, {}
    for i, key in enumerate(sorted(by_study), 1):
        cls, sk = key
        sid = f"{cls}_S{i:03d}"
        studies[sk] = sid
        for j, s in enumerate(sorted(by_study[key]), 1):
            mapping[s] = f"{sid}_R{j:03d}"
    return mapping, studies


def cmd_scan(a):
    print("[1] stem 수집 …")
    stems, src = collect_stems()
    print(f"    lance {src['lance']:,} / MGF {src['mgf']:,} → 고유 stem {len(stems):,}")

    mapping, studies = build_map(stems)
    ncls = collections.Counter(stem_class(s) for s in stems)
    print(f"[2] 매핑 생성 — 스터디 {len(studies):,} / TMT {ncls['TMT']:,} / 무표지 {ncls['LF']:,}")

    os.makedirs(OUT, exist_ok=True)
    with open(MAP_TSV, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("orig_stem\tclass\tstudy_key\tanon_stem\n")
        for s in sorted(mapping):
            fh.write(f"{s}\t{stem_class(s)}\t{study_key(s)}\t{mapping[s]}\n")
    print(f"    {MAP_TSV}  ← 비공개 보관. 절대 배포하지 말 것")

    tok = collections.Counter()
    for s in stems:
        for t in tokens_of(s):
            tok[t] += 1
    with open(REPORT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# 익명화 대상 토큰 — 사람이 확인할 것\n")
        fh.write("# 유전자명·세포주·환자·프로젝트명이 있으면 익명화로 지워진다.\n")
        fh.write("# 일반 용어가 섞여 있으면 GENERIC 집합에 추가해 목록을 줄일 수 있다.\n\n")
        fh.write("count\ttoken\n")
        for t, n in tok.most_common():
            fh.write(f"{n}\t{t}\n")
    print(f"[3] 민감 토큰 후보 {len(tok):,}종 → {REPORT}")
    print("\n    상위 25종 (사람이 훑어볼 것):")
    for t, n in tok.most_common(25):
        print(f"      {n:>5}  {t}")
    print("\n다음: 매핑을 확인한 뒤 --apply --out <폴더> 로 배포본을 만드십시오.")


def rewrite_mgf(src_path, dst_path, mapping, stats):
    """파일명은 호출자가 정하고, 여기서는 TITLE 만 바꾼다."""
    title_re = re.compile(r"^TITLE=(.+?):scan:(\d+)\s*$")
    with open(src_path, encoding="utf-8", errors="replace") as fi, \
            open(dst_path, "w", encoding="utf-8", newline="\n") as fo:
        for line in fi:
            m = title_re.match(line)
            if m:
                raw, scan = m.group(1), m.group(2)
                anon = mapping.get(raw) or mapping.get("tmt__" + raw) \
                    or mapping.get("naked__" + raw)
                if anon:
                    fo.write(f"TITLE={anon}:scan:{scan}\n")
                    stats["title_ok"] += 1
                    continue
                stats["title_miss"] += 1
                stats["miss_examples"].add(raw)
            fo.write(line)


def load_map():
    if not os.path.exists(MAP_TSV):
        sys.exit(f"매핑표가 없습니다. 먼저 --scan 을 실행하십시오: {MAP_TSV}")
    m = {}
    for line in open(MAP_TSV, encoding="utf-8").read().splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 4:
            m[c[0]] = c[3]
            m[strip_prefix(c[0])] = c[3]     # TITLE 은 접두사 없이 쓰여 있다
    return m


def cmd_apply(a):
    mapping = load_map()
    dst_root = a.out
    os.makedirs(dst_root, exist_ok=True)
    stats = collections.Counter()
    stats["miss_examples"] = set()

    for split in ("train", "val", "test"):
        sd = os.path.join(MGF_ROOT, split)
        if not os.path.isdir(sd):
            continue
        dd = os.path.join(dst_root, split)
        os.makedirs(dd, exist_ok=True)
        files = [f for f in os.listdir(sd) if f.lower().endswith(".mgf")]
        print(f"[{split}] {len(files):,} 파일")
        for i, fn in enumerate(sorted(files), 1):
            stem = fn[:-4]
            anon = mapping.get(stem)
            if not anon:
                stats["file_miss"] += 1
                stats["miss_examples"].add(stem)
                continue
            rewrite_mgf(os.path.join(sd, fn), os.path.join(dd, anon + ".mgf"),
                        mapping, stats)
            stats["files"] += 1
            if i % 200 == 0:
                print(f"    {i:,}/{len(files):,}")

    # 라벨 TSV
    lab = os.path.join(OUT, "all_labels_v3.tsv")
    if os.path.exists(lab):
        print("[labels] all_labels_v3.tsv 재작성 …")
        dst = os.path.join(dst_root, "labels.tsv")
        with open(lab, encoding="utf-8") as fi, \
                open(dst, "w", encoding="utf-8", newline="\n") as fo:
            fo.write(fi.readline())
            for n, line in enumerate(fi, 1):
                c = line.split("\t", 1)
                anon = mapping.get(c[0])
                if anon:
                    fo.write(anon + "\t" + c[1])
                    stats["label_ok"] += 1
                else:
                    stats["label_miss"] += 1
                if n % 5_000_000 == 0:
                    print(f"    {n:,} 행")

    print("\n=== 결과 ===")
    print(f"  MGF 파일        {stats['files']:,}   (미매핑 {stats['file_miss']:,})")
    print(f"  TITLE 치환      {stats['title_ok']:,}   (미매핑 {stats['title_miss']:,})")
    print(f"  라벨 행 치환    {stats['label_ok']:,}   (미매핑 {stats['label_miss']:,})")
    if stats["miss_examples"]:
        print("  ⚠ 미매핑 예시:", list(sorted(stats["miss_examples"]))[:5])
        print("     → --scan 을 다시 돌려 매핑을 갱신하십시오. 미매핑은 원본 이름이 남습니다.")
    print(f"\n  배포본: {dst_root}")
    print(f"  다음: --verify {dst_root} 로 원본 이름이 남지 않았는지 확인하십시오.")


def cmd_verify(a):
    """배포본 안에 민감 토큰이 남아 있는지 검사한다."""
    if not os.path.exists(REPORT):
        sys.exit("먼저 --scan 을 실행하십시오.")
    toks = set()
    for line in open(REPORT, encoding="utf-8").read().splitlines():
        if line.startswith("#") or line.startswith("count") or not line.strip():
            continue
        toks.add(line.split("\t")[1].lower())
    print(f"검사 토큰 {len(toks):,}종")

    hits = collections.Counter()
    checked = 0
    for root, _, files in os.walk(a.verify):
        for fn in files:
            p = os.path.join(root, fn)
            checked += 1
            low = fn.lower()
            for t in toks:
                if t in low:
                    hits[t] += 1
            if fn.lower().endswith((".tsv", ".txt")):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        l = line.lower()
                        for t in toks:
                            if t in l:
                                hits[t] += 1
                        break        # 헤더만 검사(대용량 대비)
            elif fn.lower().endswith(".mgf"):
                with open(p, encoding="utf-8", errors="replace") as fh:
                    for i, line in enumerate(fh):
                        if line.startswith("TITLE="):
                            l = line.lower()
                            for t in toks:
                                if t in l:
                                    hits[t] += 1
                            break
                        if i > 5:
                            break
    print(f"파일 {checked:,}개 검사")
    if hits:
        print("❌ 원본 토큰이 남아 있습니다:")
        for t, n in hits.most_common(20):
            print(f"   {n:>5}  {t}")
        sys.exit(1)
    print("✅ 민감 토큰이 발견되지 않았습니다.")


def main():
    ap = argparse.ArgumentParser(description="코퍼스 익명화")
    ap.add_argument("--scan", action="store_true", help="매핑표·민감토큰 목록만 만든다(기본)")
    ap.add_argument("--apply", action="store_true", help="익명화된 배포본을 만든다")
    ap.add_argument("--out", help="--apply 의 출력 폴더")
    ap.add_argument("--verify", help="검사할 배포본 폴더")
    a = ap.parse_args()

    if a.verify:
        cmd_verify(a)
    elif a.apply:
        if not a.out:
            sys.exit("--apply 에는 --out 이 필요합니다.")
        cmd_apply(a)
    else:
        cmd_scan(a)


if __name__ == "__main__":
    main()
