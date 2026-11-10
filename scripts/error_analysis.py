"""Reproduce every number in ERROR_ANALYSIS.md from the saved CSVs.

Usage (project root):
    python -m scripts.error_analysis                 # needs numpy, pandas only
    python -m scripts.error_analysis --leakage       # also the train/test overlap breakdown (needs `datasets`)
ROUGE here is an own LCS-based implementation (close to, but not identical with, the official scores
in metrics.csv); it is used for per-sample scores, bootstrap intervals and subset comparisons.
"""
import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd

from domain.services.evaluation_service import TurkishTokenizer
from domain.services.supplementary_metrics import count_unsupported
from infrastructure.csv_exporter import read_predictions

CONDS = ["C1_ZeroShot", "C2_FewShot", "C3_CoT", "C4_LoRA"]
_TOK = TurkishTokenizer()
_CJK = re.compile(r"[㐀-鿿＀-￯]")


def _lcs(a, b) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if not overlap:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_sample(pred: str, ref: str):
    p, r = _TOK.tokenize(pred), _TOK.tokenize(ref)
    if not p or not r:
        return 0.0, 0.0, 0.0
    out = []
    for n in (1, 2):
        ov = sum((_ngrams(p, n) & _ngrams(r, n)).values())
        out.append(_f1(ov, max(len(p) - n + 1, 1), max(len(r) - n + 1, 1)))
    out.append(_f1(_lcs(p, r), len(p), len(r)))
    return tuple(out)


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    d = a - b
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean() * 100, np.percentile(means, 2.5) * 100, np.percentile(means, 97.5) * 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="outputs/predictions.csv")
    ap.add_argument("--raw", default="outputs/raw_generations.csv")
    ap.add_argument("--leakage", action="store_true")
    ap.add_argument("--n-train", type=int, default=1000, help="training size of the run (for --leakage)")
    args = ap.parse_args()
    df = read_predictions(args.predictions)
    raw = pd.read_csv(args.raw, keep_default_na=False)
    col = {c: f"pred_{c}" for c in CONDS}
    refs, arts = df["reference_summary"].tolist(), df["news_text"].tolist()

    print("== Own ROUGE (mean %, R-1 / R-2 / R-L) ==")
    per = {}
    for c in CONDS:
        per[c] = np.array([rouge_sample(p, r) for p, r in zip(df[col[c]], refs)])
        print(f"{c:12s} {per[c][:, 0].mean()*100:6.2f} {per[c][:, 1].mean()*100:6.2f} {per[c][:, 2].mean()*100:6.2f}")

    print("\n== Paired bootstrap on per-sample ROUGE-L (diff, 95% CI) ==")
    for a, b in [("C4_LoRA", "C1_ZeroShot"), ("C4_LoRA", "C2_FewShot"), ("C4_LoRA", "C3_CoT"),
                 ("C2_FewShot", "C1_ZeroShot"), ("C3_CoT", "C1_ZeroShot")]:
        m, lo, hi = boot_diff(per[a][:, 2], per[b][:, 2])
        print(f"{a}-{b}: {m:+.1f} [{lo:.1f}, {hi:.1f}]")

    print("\n== A. Extractive vs abstractive ==")
    def copied_3g(texts):
        found = total = 0
        for t, a in zip(texts, arts):
            g, ag = _ngrams(_TOK.tokenize(t), 3), set(_ngrams(_TOK.tokenize(a), 3))
            found += sum(v for k, v in g.items() if k in ag); total += sum(g.values())
        return found / total if total else 0
    def novel_uni(texts):
        nov = tot = 0
        for t, a in zip(texts, arts):
            av = set(_TOK.tokenize(a)); tk = _TOK.tokenize(t)
            nov += sum(1 for w in tk if w not in av); tot += len(tk)
        return nov / tot if tot else 0
    rows = {"Ref": refs, **{c: df[col[c]].tolist() for c in CONDS}}
    for name, texts in rows.items():
        words = np.mean([len(t.split()) for t in texts])
        ell = sum(t.rstrip().endswith("...") for t in texts)
        print(f"{name:12s} words={words:5.1f}  3gram-in-article={copied_3g(texts):.3f}  "
              f"novel-unigrams={novel_uni(texts):.3f}  ends-with-'...'={ell}")

    print("\n== B. Entities / language drift ==")
    for c in CONDS:
        flagged = sum(count_unsupported(p, a)[0] > 0 for p, a in zip(df[col[c]], arts))
        cjk = sum(bool(_CJK.search(p)) for p in df[col[c]])
        print(f"{c:12s} samples with >=1 unsupported entity/number: {flagged:3d}   CJK outputs: {cjk}")

    print("\n== C. Degenerate generation (raw outputs) ==")
    for c in CONDS:
        r = raw[f"raw_{c}"].tolist()
        loops = sum(any(v >= 3 for v in _ngrams(t.split(), 3).values()) for t in r)
        def has_repeat(t):
            s = [x.strip() for x in re.split(r"(?<=[.!?…])\s+", t) if x.strip()]
            return len(s) != len(set(s))
        trunc = sum(not t.rstrip().endswith((".", "!", "?", "…", "...", '"', "”")) for t in r)
        extra = ""
        if c == "C3_CoT":
            extra = f"   no 'Nihai özet' marker: {sum(not re.search(r'n[iİI]ha[iİI]', t, re.I) for t in r)}"
        print(f"{c:12s} trigram loops: {loops}  repeated sentences: {sum(map(has_repeat, r))}  "
              f"no terminal punctuation: {trunc}{extra}")

    if args.leakage:
        from infrastructure.dataset_repository import TurkishNewsDatasetRepository
        from scripts.check_leakage import find_overlaps
        train, test = TurkishNewsDatasetRepository(42).load_splits(args.n_train, len(df))
        assert [a.id for a in test] == df["sample_id"].tolist(), "test order differs from predictions CSV"
        art = find_overlaps([a.article_text for a in train], [a.article_text for a in test], 0.5)
        flagged = sorted({j for hits in art.values() for j, _, _ in hits})
        template = [j for j in flagged if "kim kime dum duma" in arts[j].lower()]
        topical = [j for j in flagged if "sarı serum" in arts[j].lower() and j not in template]
        same_event = [j for j in flagged if j not in template and j not in topical]
        groups = {f"all {len(df)}": list(range(len(df))), f"flagged ({len(flagged)})": flagged,
                  f"template ({len(template)})": template, f"same-event ({len(same_event)})": same_event,
                  f"topical ({len(topical)})": topical,
                  "without template": [j for j in range(len(df)) if j not in template],
                  "without same-event": [j for j in range(len(df)) if j not in same_event],
                  "without all flagged": [j for j in range(len(df)) if j not in flagged]}
        print("\n== Leakage subsets: mean own ROUGE-L (%) ==")
        print(f"{'':22s}" + "".join(f"{c:>13s}" for c in CONDS))
        for g, idx in groups.items():
            print(f"{g:22s}" + "".join(f"{per[c][idx, 2].mean()*100:13.1f}" for c in CONDS))


if __name__ == "__main__":
    main()
