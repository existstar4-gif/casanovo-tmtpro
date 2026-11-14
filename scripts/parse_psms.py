"""
parse_psms.py  —  TMTpro de novo fine-tuning 데이터 준비 (1단계: PSM -> 정답 라벨)

[v2 설계 — 실데이터 감사 반영]
이 데이터셋에서는 검색이 static/dynamic 이든 Modifications 컬럼에 TMT 가 ~100%
기록됨이 확인됨(STATIC/HIDDEN-STATIC=100%, 정상 DYNAMIC=98%+). 따라서:
  - TMT 를 '주입' 하지 않고 Modifications 컬럼을 그대로 신뢰한다.
  - 파일 단위 'TMT 비율 게이트' 로 비-TMT(label-free, 예: <sample>) 파일을 자동 제외.
  - 한 PSM 파일이 여러 fraction(File ID F.1,F.2,...)을 담으므로 키는 (raw_stem, scan).
    raw_stem 은 File ID -> InputFiles.txt 매핑(라인별)으로 fraction 을 구분.
  - 다중 PD 재분석(479 raws)으로 같은 (raw,scan) 이 다른 서열로 동정되는 충돌(~9%)은
    drop (서열 일치 시에는 q-value 최저 1건 채택).

usage:
  python parse_psms.py PSMS.txt [...] -o out.tsv
  python parse_psms.py --glob "data/pd/**/*_PSMs.txt" -o all.tsv
출력: raw_stem, scan, charge, proforma, peptide
"""
from __future__ import annotations
import argparse
import glob as globmod
import os
import re
import sys
from collections import Counter

TMTPRO_MASS = 304.207146

RESIDUE_MOD_MAP = {
    "Oxidation": "Oxidation",
    "Carbamidomethyl": "Carbamidomethyl",
    "Phospho": "Phospho",
    "Deamidated": "Deamidated",
    "TMTpro": "TMTpro",
}
NTERM_MOD_MAP = {
    "Acetyl": "Acetyl",
    "Met-loss+Acetyl": "Acetyl",   # 개시 Met 제거는 Annotated Sequence 에 이미 반영됨
    "Met-loss": None,
    "Carbamyl": "Carbamyl",
    "TMTpro": "TMTpro",
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
RE_RESIDUE = re.compile(r"^([A-Za-z])(\d+)\((.+)\)$")
RE_TERM = re.compile(r"^([NC])-Term(?:\(Prot\))?\((.+)\)$", re.IGNORECASE)


def clean_field(s: str) -> str:
    return s.strip().strip('"').strip()


def extract_peptide(annotated_seq: str) -> str | None:
    s = clean_field(annotated_seq)
    m = re.match(r"^\[[^\]]*\]\.(.+)\.\[[^\]]*\]$", s)
    if m:
        pep = m.group(1)
    elif "." in s:
        parts = s.split(".")
        pep = parts[1] if len(parts) >= 3 else s
    else:
        pep = s
    pep = pep.upper()
    if not pep or any(c not in VALID_AA for c in pep):
        return None
    return pep


def parse_mods(mod_field: str):
    """ -> residue_mods{pos:name}, nterm, cterm, unknown[], met_loss(bool) """
    residue_mods: dict[int, str] = {}
    nterm = None
    cterm = None
    unknown = []
    met_loss = False
    field = clean_field(mod_field)
    if not field:
        return residue_mods, nterm, cterm, unknown, met_loss
    for raw in field.split(";"):
        tok = raw.strip()
        if not tok:
            continue
        mt = RE_TERM.match(tok)
        if mt:
            term, name = mt.group(1).upper(), mt.group(2).strip()
            name = name.split(")(")[-1].strip()
            if "Met-loss" in name:           # 개시 Met 제거 (시퀀스에서 M 떼야 함)
                met_loss = True
            mapped = NTERM_MOD_MAP.get(name, "UNMAPPED")
            if mapped == "UNMAPPED":
                unknown.append(tok)
            elif mapped is not None:
                if term == "N":
                    nterm = mapped
                else:
                    cterm = mapped
            continue
        mr = RE_RESIDUE.match(tok)
        if mr:
            pos = int(mr.group(2))
            name = mr.group(3).strip()
            mapped = RESIDUE_MOD_MAP.get(name)
            if mapped is None:
                unknown.append(tok)
            else:
                residue_mods[pos] = mapped
            continue
        unknown.append(tok)
    return residue_mods, nterm, cterm, unknown, met_loss


def build_proforma(peptide: str, residue_mods: dict[int, str], nterm: str | None) -> str:
    """
    Modifications 컬럼을 그대로 신뢰해 렌더링 (TMT 주입하지 않음).

    근거(backward 전구체-질량 검증으로 확정): TMT 라벨링은 100% 가 아니며, dynamic 검색은
    각 PSM 의 *실제* 라벨 상태를 보고한다. 빈 N-term/K 는 'annotation 누락'이 아니라
    '진짜 무라벨'인 경우가 있어, 주입하면 전구체 질량이 +304 Da 어긋난다(틀린 라벨).
    따라서 컬럼에 명시된 mod 만 반영한다.
    """
    out = []
    if nterm:
        out.append(f"[{nterm}]-")
    for i, aa in enumerate(peptide, start=1):
        if i in residue_mods:
            out.append(f"{aa}[{residue_mods[i]}]")
        else:
            out.append(aa)
    return "".join(out)


def load_inputfiles_map(psm_path: str) -> dict[str, str]:
    base = psm_path
    if base.endswith("_PSMs.txt"):
        base = base[: -len("_PSMs.txt")]
    inp = base + "_InputFiles.txt"
    mapping: dict[str, str] = {}
    if not os.path.exists(inp):
        return mapping
    with open(inp, encoding="utf-8", errors="replace") as fh:
        header = [clean_field(c) for c in fh.readline().rstrip("\n").split("\t")]
        try:
            i_id = header.index("Study File ID")
            i_name = header.index("File Name")
        except ValueError:
            return mapping
        for line in fh:
            cells = line.rstrip("\n").split("\t")
            if len(cells) <= max(i_id, i_name):
                continue
            fid = clean_field(cells[i_id])
            fname = clean_field(cells[i_name])
            if not fid or not fname:
                continue
            stem = os.path.splitext(os.path.basename(fname.replace("\\", "/")))[0]
            mapping[fid] = stem
    return mapping


def file_tmt_fraction(psm_path: str, limit: int = 8000) -> float | None:
    """ Modifications 컬럼에 TMTpro 가 등장하는 행 비율 (TMT 검색 여부 게이트용). """
    n = hit = 0
    with open(psm_path, encoding="utf-8", errors="replace") as fh:
        header = [clean_field(c) for c in fh.readline().split("\t")]
        if "Modifications" not in header:
            return None
        mi = header.index("Modifications")
        for line in fh:
            c = line.split("\t")
            if len(c) <= mi:
                continue
            n += 1
            if "TMTpro" in c[mi]:
                hit += 1
            if n >= limit:
                break
    return (hit / n) if n else None


def process_file(psm_path: str, q_thresh: float, stats: Counter):
    """ yield (key=(raw_stem,scan), charge, proforma, peptide, q_value) — dedup 은 호출부에서. """
    fid_map = load_inputfiles_map(psm_path)
    psm_base = os.path.basename(psm_path)
    with open(psm_path, encoding="utf-8", errors="replace") as fh:
        header = [clean_field(c) for c in fh.readline().rstrip("\n").split("\t")]
        idx = {name: i for i, name in enumerate(header)}

        def col(cells, name, default=""):
            i = idx.get(name)
            return clean_field(cells[i]) if (i is not None and i < len(cells)) else default

        for line in fh:
            cells = line.rstrip("\n").split("\t")
            stats["rows"] += 1

            if col(cells, "Confidence") != "High":
                stats["drop_conf"] += 1
                continue
            qv = col(cells, "q-Value")
            q_num = 1.0
            try:
                if qv != "":
                    q_num = float(qv)
                    if q_num >= q_thresh:
                        stats["drop_q"] += 1
                        continue
            except ValueError:
                pass
            rank = col(cells, "Search Engine Rank") or col(cells, "Rank")
            if rank not in ("", "1"):
                stats["drop_rank"] += 1
                continue

            peptide = extract_peptide(col(cells, "Annotated Sequence"))
            if peptide is None:
                stats["drop_seq"] += 1
                continue

            rmods, nterm, cterm, unknown, met_loss = parse_mods(col(cells, "Modifications"))
            if unknown:
                stats["drop_unkmod"] += 1
                continue

            # 개시 Met 제거: 시퀀스에서 앞 M 떼고 잔기 mod 위치를 -1 시프트
            if met_loss and peptide.startswith("M"):
                peptide = peptide[1:]
                rmods = {p - 1: m for p, m in rmods.items() if p >= 2}
                if not peptide:
                    stats["drop_seq"] += 1
                    continue

            charge = col(cells, "Charge")
            scan = col(cells, "First Scan")
            file_id = col(cells, "File ID")
            prec_mz = col(cells, "m/z [Da]")    # PD 보고 전구체 (검증된 monoisotopic)
            if not scan or not charge:
                stats["drop_scan"] += 1
                continue
            # raw_stem: File ID -> fraction raw. fallback 은 PSM파일명에 묶어 교차충돌 방지.
            if file_id in fid_map:
                raw_stem = fid_map[file_id]
            elif len(fid_map) == 1:
                raw_stem = next(iter(fid_map.values()))
            else:
                raw_stem = f"{psm_base}#{file_id}"

            proforma = build_proforma(peptide, rmods, nterm)
            yield (raw_stem, scan), charge, proforma, peptide, q_num, prec_mz


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("psms", nargs="*")
    ap.add_argument("--glob")
    ap.add_argument("--filelist", help="text file with one PSMs path per line")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-q", "--q-thresh", type=float, default=0.01)
    ap.add_argument("--tmt-gate", type=float, default=0.5,
                    help="파일 TMT 비율이 이 값 미만이면 비-TMT 로 보고 제외")
    ap.add_argument("--naked", action="store_true",
                    help="INVERT gate: keep NON-TMT (label-free) files only -> naked labels")
    ap.add_argument("--conflict", choices=["drop", "bestscore"], default="drop")
    args = ap.parse_args(argv)

    files = list(args.psms)
    if args.glob:
        files += globmod.glob(args.glob, recursive=True)
    if args.filelist:
        with open(args.filelist, encoding="utf-8") as fh:
            files += [ln.strip() for ln in fh if ln.strip()]
    files = [f for f in files if f.lower().endswith("_psms.txt")]
    if not files:
        print("no PSM files given", file=sys.stderr)
        return 2

    stats = Counter()
    # 키별 최적 후보: key -> [best_q, charge, proforma, plain_pep, conflict_flag]
    best: dict[tuple, list] = {}

    for fp in files:
        stats["files"] += 1
        frac = file_tmt_fraction(fp)
        if args.naked:
            # naked mode: keep NON-TMT files (low TMT fraction); skip TMT files
            if frac is not None and frac >= args.tmt_gate:
                stats["files_excluded_TMT"] += 1
                continue
        else:
            if frac is None or frac < args.tmt_gate:
                stats["files_excluded_nonTMT"] += 1
                continue
        stats["files_tmt"] += 1
        for key, charge, proforma, peptide, q_num, prec_mz in process_file(fp, args.q_thresh, stats):
            cur = best.get(key)
            if cur is None:
                best[key] = [q_num, charge, proforma, peptide, False, prec_mz]
            else:
                if cur[3] != peptide:
                    cur[4] = True          # 서열 불일치 -> 충돌 표시
                if q_num < cur[0]:         # 더 좋은(낮은) q 면 교체
                    cur[0], cur[1], cur[2], cur[3], cur[5] = q_num, charge, proforma, peptide, prec_mz

    n_out = n_conflict = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        out.write("raw_stem\tscan\tcharge\tprec_mz\tproforma\tpeptide\n")
        for (raw_stem, scan), (q, charge, proforma, peptide, conflict, prec_mz) in best.items():
            if conflict and args.conflict == "drop":
                n_conflict += 1
                continue
            out.write(f"{raw_stem}\t{scan}\t{charge}\t{prec_mz}\t{proforma}\t{peptide}\n")
            n_out += 1

    print(f"[done] files={stats['files']} (TMT={stats['files_tmt']} "
          f"excluded_nonTMT={stats['files_excluded_nonTMT']}) rows={stats['rows']}",
          file=sys.stderr)
    print(f"   unique_keys={len(best)}  conflicts_dropped={n_conflict}  written={n_out}",
          file=sys.stderr)
    for k in ("drop_conf", "drop_q", "drop_rank", "drop_seq", "drop_unkmod", "drop_scan"):
        if stats[k]:
            print(f"   {k}={stats[k]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
