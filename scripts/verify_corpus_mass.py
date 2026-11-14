"""
verify_corpus_mass.py — 전체 라벨 코퍼스 질량 정합성 검사 (스펙트럼 불필요).

각 라벨 행에 대해:
  이론 전구체 중성질량(proforma 로 계산) vs 관측 중성질량(prec_mz*z - z*proton)
  -> ppm 오차. PD 의 prec_mz 는 고정밀 monoisotopic 이므로, 전 코퍼스에서
     |ppm| 이 작으면 proforma 구성(TMT 배치·Met-loss·mod 파싱)이 질량 정합적임을 증명.

verify_match.py 와 달리 mzML 없이 7.66M 라벨 '전체'를 검사한다.

usage: python scripts/verify_corpus_mass.py out/all_labels_v2.tsv
"""
from __future__ import annotations
import argparse, re, sys

PROTON = 1.0072764665
H2O = 18.0105646863
AA = {
    "G":57.02146372,"A":71.03711379,"S":87.03202841,"P":97.05276385,"V":99.06841392,
    "T":101.04767847,"C":103.00918448,"L":113.08406398,"I":113.08406398,"N":114.04292745,
    "D":115.02694303,"Q":128.05857751,"K":128.09496302,"E":129.04259309,"M":131.04048509,
    "H":137.05891186,"F":147.06841392,"R":156.10111102,"Y":163.06332854,"W":186.07931295,
}
MOD = {
    "TMTpro":304.2071453,"Oxidation":15.9949146,"Carbamidomethyl":57.0214637,
    "Phospho":79.9663305,"Acetyl":42.0105646,"Deamidated":0.9840156,"Carbamyl":43.0058578,
}


def proforma_mass(seq: str) -> float:
    nterm = 0.0
    m = re.match(r"^\[([^\]]+)\]-(.*)$", seq)
    if m:
        nterm = MOD[m.group(1)]
        seq = m.group(2)
    total = nterm + H2O
    i = 0
    while i < len(seq):
        aa = seq[i]; i += 1
        total += AA[aa]
        if i < len(seq) and seq[i] == "[":
            j = seq.index("]", i)
            total += MOD[seq[i+1:j]]
            i = j + 1
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--limit", type=int, default=0, help="0 = 전체")
    args = ap.parse_args()

    # ppm 히스토그램 버킷
    buckets = [1, 2, 5, 10, 20, 50]
    counts = {b: 0 for b in buckets}
    counts["over"] = 0
    n = 0; n_noprec = 0; n_err = 0
    worst = []  # (abs_ppm, line) top offenders

    with open(args.tsv, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {name: i for i, name in enumerate(header)}
        ci_charge, ci_pm, ci_prof = col["charge"], col["prec_mz"], col["proforma"]
        for line in fh:
            c = line.rstrip("\n").split("\t")
            pm = c[ci_pm]
            if not pm:
                n_noprec += 1; continue
            try:
                z = int(c[ci_charge]); prec = float(pm)
                th = proforma_mass(c[ci_prof])
            except (KeyError, ValueError):
                n_err += 1; continue
            obs = prec * z - z * PROTON
            ppm = abs((obs - th) / th * 1e6)
            for b in buckets:
                if ppm < b:
                    counts[b] += 1
                    break
            else:
                counts["over"] += 1
                if len(worst) < 20:
                    worst.append((ppm, c[ci_prof], z, prec))
            n += 1
            if args.limit and n >= args.limit:
                break

    print(f"검사한 라벨: {n:,}  (prec_mz 없음 {n_noprec:,} / 파싱오류 {n_err:,})")
    print("-" * 50)
    cum = 0
    for b in buckets:
        cum += counts[b]
        print(f"  |ppm| < {b:>3} : {counts[b]:>10,}   누적 {100*cum/n:6.3f}%")
    print(f"  |ppm| >=50 : {counts['over']:>10,}   ({100*counts['over']/n:.4f}%)")
    print("-" * 50)
    within10 = sum(counts[b] for b in (1, 2, 5, 10))
    print(f"==> |ppm| < 10 (PD 고정밀 일치): {within10:,}/{n:,} = {100*within10/n:.4f}%")
    if worst:
        print("\n상위 이탈 샘플(>=50ppm):")
        for p, prof, z, prec in sorted(worst, reverse=True)[:10]:
            print(f"  {p:8.1f}ppm  z{z}  prec={prec:.4f}  {prof[:55]}")


if __name__ == "__main__":
    main()
