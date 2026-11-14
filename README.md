# Casanovo-TMT

Code accompanying *Casanovo-TMT: deep learning-based de novo sequencing of TMTpro-labeled peptides and phosphopeptides* (Kim, H. B. and Kim, K.-e.).

Casanovo-TMT is [Casanovo](https://github.com/Noble-Lab/casanovo) 5.2.0 fine-tuned on TMTpro-labeled
spectra, with the residue vocabulary extended by the TMTpro tags and phosphorylation, and with
precursor-mass re-ranking of the candidates held in the beam.

This repository contains code and configuration only. It contains no mass spectrometry data and no
model weights.

## Contents

| Path | What it is |
|---|---|
| `casanovo_patch/` | Changes to Casanovo 5.2.0: unified diffs against the PyPI release, the three modified files, their SHA-256 checksums and an install script |
| `configs/casanovo_tmt_train.yaml` | Fine-tuning configuration (beam width 1, used during training only) |
| `configs/casanovo_tmt_inference.yaml` | Inference configuration (beam width 10, used for every reported result) |
| `scripts/` | Corpus assembly, training-set interleaving, evaluation and figure scripts |
| `scripts/quant/` | Analysis of the phosphoproteomic studies (Fig. 4 and 5) |
| `requirements.lock.txt` | Package versions of the environment that produced the results |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install casanovo==5.2.0
python casanovo_patch/apply_patch.py
```

`apply_patch.py` replaces `denovo/model.py`, `denovo/model_runner.py` and `data/ms_io.py` in the
installed package and verifies their checksums. The results were produced with Python 3.12 and
PyTorch 2.11 (CUDA 12.8); `requirements.lock.txt` lists every package version.

## Source changes

1. **Extended vocabulary.** Rows for the new tokens (`[TMTpro]-`, `K[TMTpro]` and the
   phosphorylation tokens) are added to the decoder of the released checkpoint and initialized by
   copying the rows of existing residues; the rows of existing tokens are kept.
2. **N-terminal modifications in beam search.** Casanovo 5.2.0 discards beams that end in an
   N-terminal modification during reverse decoding, so no N-terminally modified peptide, including
   every TMTpro-labeled peptide, can be predicted. The patch restores these beams; the defect is
   fixed upstream in 5.2.1.
3. **Precursor-mass re-ranking.** With `CASANOVO_KNAPSACK=1`, candidates in the beam whose
   peptidoform mass matches the measured precursor within 20 ppm (`CASANOVO_KNAPSACK_PPM`), allowing
   one carbon-13 isotope, are ranked above the others. No candidate is discarded.
4. **Tryptic search-space re-ranking.** With `CASANOVO_DBRANK=1` and
   `CASANOVO_DBRANK_INDEX=<index.npy>`, candidates present in a tryptic digest of a reference
   proteome are additionally ranked first within each mass group. Build the index with
   `scripts/build_tryptic_index.py --fasta <proteome.fasta> -o <index.npy>`.
5. **Run settings in the output.** The values of these switches are written to the mzTab metadata
   of every run.

All switches are off by default, so the patched package behaves as Casanovo 5.2.0 unless they are
set. Other switches in `model.py` (`CASANOVO_GATE*`, `CASANOVO_TMT_MARGIN`, `CASANOVO_TMT_LAMBDA`)
belong to experiments that were not used for the reported results and should be left unset.

## Running

Prediction with precursor-mass re-ranking, the default setting of the paper:

```bash
export CASANOVO_KNAPSACK=1
casanovo sequence -c configs/casanovo_tmt_inference.yaml -m <casanovo_tmt.ckpt> -o out.mztab spectra.mgf
```

Fine-tuning starts from the released Casanovo 5.2.0 weights:

```bash
casanovo train -c configs/casanovo_tmt_train.yaml -m <casanovo_orbitrap_v5-2-0.ckpt> \
    -p <validation.mgf> <training.mgf ...>
```

The training rows must be interleaved so that label-free and TMTpro-labeled spectra are mixed within
every epoch (`scripts/shuffle_train_lance.py`). Training on a table ordered by file, in which the
spectra of each class form one block, erases the label-free capability of the model.

## Paths

The scripts were run on local data. Data locations are written as relative paths under `data/`
(spectra, search results and evaluation sets) and `out/` (intermediate results). The studies are
named Study 1 to 7 as in the paper.

## License

Apache License 2.0 (see `LICENSE`). The files in `casanovo_patch/modified/` are modified versions of
files from Casanovo 5.2.0 (Copyright Noble Lab, University of Washington, Apache License 2.0); the
changes are given as diffs in `casanovo_patch/`.

## Citation

Kim, H. B. and Kim, K.-e. Casanovo-TMT: deep learning-based de novo sequencing of TMTpro-labeled peptides and phosphopeptides. *bioRxiv* (2026).
