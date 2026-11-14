# -*- coding: utf-8 -*-
"""Compute AA-level precision/recall + peptide precision the SAME way Casanovo does
(casanovo.denovo.evaluate.aa_match_batch), so our numbers are comparable to the
0.95 AA-level average precision Casanovo reports on within-distribution data.
Filters to a spectrum class (tmt/naked). Uses mass-based AA matching (I/L equal).
Usage: python aa_eval.py <mztab> <gt.tsv> <class: tmt|naked|all>
"""
import sys, re
from casanovo.denovo.evaluate import aa_match_batch, aa_match_metrics

mztab, gtf, cls_want = sys.argv[1], sys.argv[2], sys.argv[3]

# token -> mass (from casanovo_tmt_v4.yaml residues)
AA = {
 "G":57.021464,"A":71.037114,"S":87.032028,"P":97.052764,"V":99.068414,
 "T":101.047670,"C[Carbamidomethyl]":160.030649,"L":113.084064,"I":113.084064,
 "N":114.042927,"D":115.026943,"Q":128.058578,"K":128.094963,"E":129.042593,
 "M":131.040485,"H":137.058912,"F":147.068414,"R":156.101111,"Y":163.063329,
 "W":186.079313,"M[Oxidation]":147.035400,"N[Deamidated]":115.026943,
 "Q[Deamidated]":129.042594,"[Acetyl]-":42.010565,"[Carbamyl]-":43.005814,
 "[Ammonia-loss]-":-17.026549,"[+25.980265]-":25.980265,"S[Phospho]":166.998359,
 "T[Phospho]":181.014001,"Y[Phospho]":243.029660,"K[TMTpro]":432.302109,
 "[TMTpro]-":304.207146,"R[Phospho]":236.067442,"K[Phospho]":208.061294,
}
TOKENS = sorted(AA.keys(), key=len, reverse=True)  # longest-match-first

def tok(seq):
    seq = seq.strip()
    out, i, n = [], 0, len(seq)
    while i < n:
        m = None
        for t in TOKENS:
            if seq.startswith(t, i):
                m = t; break
        if m is None:
            i += 1; continue
        out.append(m); i += len(m)
    return out

# GT
gt = {}
with open(gtf, encoding="utf-8") as fh:
    fh.readline()
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            gt[int(p[0])] = (p[1], p[2])

# predictions
idx_re = re.compile(r"index=(\d+)")
header = None
preds = {}
with open(mztab, encoding="utf-8") as fh:
    for line in fh:
        if line.startswith("PSH"):
            header = line.rstrip("\n").split("\t"); continue
        if not line.startswith("PSM"): continue
        c = dict(zip(header, line.rstrip("\n").split("\t")))
        m = idx_re.search(c.get("spectra_ref", ""))
        if not m: continue
        pro = c.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence") or c.get("sequence","")
        preds[int(m.group(1))] = pro

# build aligned lists (only spectra of wanted class that have a prediction)
truths, guesses = [], []
n_total = 0
for i,(cl,seq) in gt.items():
    if cls_want != "all" and cl != cls_want: continue
    n_total += 1
    if i in preds and preds[i]:
        truths.append(tok(seq))
        guesses.append(tok(preds[i]))

matches, n_aa_true, n_aa_pred = aa_match_batch(truths, guesses, AA)
aa_prec, aa_rec, pep_prec = aa_match_metrics(matches, n_aa_true, n_aa_pred)
n_emit = len(guesses)
print(f"=== {mztab.split('/')[-1]}  class={cls_want} ===")
print(f"  spectra: {n_total} (emitted {n_emit}, coverage {100*n_emit/n_total:.1f}%)")
print(f"  AA-level precision : {aa_prec:.4f}   (Casanovo within-dist ~0.95)")
print(f"  AA-level recall    : {aa_rec:.4f}")
print(f"  peptide precision  : {pep_prec:.4f}   (exact full-sequence match)")
