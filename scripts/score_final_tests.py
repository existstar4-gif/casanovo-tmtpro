# -*- coding: utf-8 -*-
"""score_final_tests.py — 설정 결정 이후 평가셋(test 2 = 홀드아웃, test 3, test 4) 채점 (2026-11-10).

채점 규칙은 test 1(sample_new)과 같다: 완전 펩타이도폼 일치, I=L, 분모 = 정답이 있는 전체
스펙트럼(예측 없음 = 오답). 지표는 PREREG_holdout.md 가 지정한 것만 낸다.
  - 커버리지 100% 정밀도 (TMT / naked) + 펩타이드 클러스터 부트스트랩 95% CI (10,000회)
  - 오류 <=5% / <=1% 를 만족하는 최대 커버리지 (점수 순)
  - 계층: TMT 비인산화 / 인산화 1 / 인산화 2+ / naked,  길이 6-9 / 10-12 / 13-15 / 16-19 / 20+
  - naked 스펙트럼에 TMTpro 토큰을 낸 비율
사용: python scripts/score_final_tests.py <gt.tsv> <mztab> [<label>]
      python scripts/score_final_tests.py --check   (test 1 로 84.7 / 61.2 재현 확인)
"""
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")


def full(s):
    return s.strip().replace("I", "L")


def load_gt(path):
    gt = {}
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 3:
            gt[int(p[0])] = (p[1], p[2])
    return gt


def load_mztab(path):
    out, hdr = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("PSH"):
            hdr = line.rstrip("\n").split("\t")
            continue
        if not line.startswith("PSM") or not hdr:
            continue
        c = dict(zip(hdr, line.rstrip("\n").split("\t")))
        m = re.search(r"index=(\d+)", c.get("spectra_ref", ""))
        if not m:
            continue
        seq = c.get("opt_global_cv_MS:1003169_proforma_peptidoform_sequence") or c.get("sequence", "")
        try:
            sc = float(c.get("search_engine_score[1]", 0))
        except ValueError:
            sc = 0.0
        out[int(m.group(1))] = (seq, sc)
    return out


def length(seq):
    return len(re.findall(r"[A-Z]", re.sub(r"\[[^\]]*\]", "", seq)))


def lbin(n):
    return "6-9" if n <= 9 else "10-12" if n <= 12 else "13-15" if n <= 15 else "16-19" if n <= 19 else "20+"


def cluster_ci(rows, n_boot=10000, seed=1):
    """rows = [(peptide, correct)] — 펩타이드 단위로 재표집."""
    by = {}
    for pep, ok in rows:
        by.setdefault(full(pep), []).append(ok)
    keys = list(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        k = s = 0
        for _ in keys:
            v = by[keys[rng.randrange(len(keys))]]
            k += sum(v)
            s += len(v)
        vals.append(100 * k / s)
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def max_cov(items, n_total, err):
    """items = [(score, correct)] 예측된 것만. 분모는 전체 스펙트럼."""
    items = sorted(items, key=lambda x: -x[0])
    best, k = 0, 0
    for i, (_, ok) in enumerate(items, 1):
        k += ok
        if k / i >= 1 - err:
            best = i
    return 100 * best / n_total


def score(gt_path, mz_path, label="", ci=True):
    gt, pred = load_gt(gt_path), load_mztab(mz_path)
    res = {}
    for cls in ("tmt", "naked"):
        idx = [i for i, (c, _) in gt.items() if c == cls]
        rows, items, strata, lens, fp = [], [], {}, {}, 0
        for i in idx:
            ref = gt[i][1]
            p = pred.get(i)
            ok = bool(p) and full(p[0]) == full(ref)
            rows.append((ref, ok))
            if p:
                items.append((p[1], ok))
                if cls == "naked" and "TMTpro" in p[0]:
                    fp += 1
            nph = ref.count("[Phospho]")
            st = "nonphos" if nph == 0 else "phos1" if nph == 1 else "phos2+"
            strata.setdefault(st, []).append(ok)
            lens.setdefault(lbin(length(ref)), []).append(ok)
        n = len(idx)
        prec = 100 * sum(ok for _, ok in rows) / n
        lo, hi = cluster_ci(rows) if ci else (float("nan"),) * 2
        res[cls] = dict(n=n, prec=prec, lo=lo, hi=hi, cov5=max_cov(items, n, 0.05),
                        cov1=max_cov(items, n, 0.01), predicted=100 * len(items) / n,
                        strata={k: (len(v), 100 * sum(v) / len(v)) for k, v in strata.items()},
                        lens={k: (len(v), 100 * sum(v) / len(v)) for k, v in lens.items()},
                        fp=100 * fp / max(1, len(items)))
    print(f"=== {label or mz_path} ===")
    for cls, r in res.items():
        print(f"  {cls:5s} n={r['n']:5d}  precision {r['prec']:.1f}%  CI {r['lo']:.1f}-{r['hi']:.1f}  "
              f"predicted {r['predicted']:.1f}%  max coverage @<=5% err {r['cov5']:.1f}%  @<=1% {r['cov1']:.1f}%")
        print("        strata " + "  ".join(f"{k} {v[1]:.1f}% (n={v[0]})" for k, v in sorted(r["strata"].items())))
        print("        length " + "  ".join(f"{k} {r['lens'][k][1]:.1f}% (n={r['lens'][k][0]})"
                                         for k in ("6-9", "10-12", "13-15", "16-19", "20+") if k in r["lens"]))
        if cls == "naked":
            print(f"        TMTpro token on naked predictions {r['fp']:.2f}%")
    return res


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        score("data/eval/sample_new_gt.tsv",
              "data/eval/ladder_check/final_knap_recheck.mztab", "test 1 check", ci=False)
    else:
        score(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
