"""Trivial extractive baselines scored with the same metrics as C1-C4 (no model, runs on CPU).

Usage (project root):
    python -m scripts.baselines --predictions outputs/predictions.csv --n-train 4907
Baselines: the first sentence(s) of the article, a reference summary of another article (a floor for the
metrics), and its first N words, where N is the mean reference length of the
training split (--n-train rows), so no test reference is used to choose N. Without --n-train, --words is used.
"""
import argparse
import os
import re
from typing import List

import pandas as pd

from domain.services.evaluation_service import EvaluationService
from domain.services.supplementary_metrics import SupplementaryEvaluationService
from infrastructure.csv_exporter import read_predictions

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def lead_sentences(text: str, k: int) -> str:
    return " ".join([s for s in _SENTENCE_SPLIT.split(text.strip()) if s][:k])


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def shifted_references(references: List[str]) -> List[str]:
    """Reference of the next sample (cyclic): a fluent summary that belongs to a different article."""
    return list(references[1:]) + list(references[:1])


def mean_reference_words(references: List[str]) -> int:
    return round(sum(len(r.split()) for r in references) / len(references))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="outputs/predictions.csv")
    p.add_argument("--output", default=None, help="Default: baselines.csv next to the predictions file")
    p.add_argument("--n-train", type=int, default=None, help="Choose N from the mean reference length of this many train rows")
    p.add_argument("--words", type=int, default=30)
    args = p.parse_args()
    output = args.output or os.path.join(os.path.dirname(args.predictions), "baselines.csv")

    frame = read_predictions(args.predictions)
    articles, refs = frame["news_text"].tolist(), frame["reference_summary"].tolist()
    n_words = args.words
    if args.n_train:
        from infrastructure.dataset_repository import TurkishNewsDatasetRepository
        train, _ = TurkishNewsDatasetRepository(42).load_splits(args.n_train, 1)
        n_words = mean_reference_words([a.reference_summary for a in train])
    print(f"[INFO] first-N-words baseline uses N={n_words}")

    baselines = {
        "Lead-1": [lead_sentences(a, 1) for a in articles],
        "Lead-2": [lead_sentences(a, 2) for a in articles],
        "Other-article reference": shifted_references(refs),
        f"First-{n_words}-words": [first_words(a, n_words) for a in articles],
    }
    rouge_service, extra_service = EvaluationService(), SupplementaryEvaluationService()
    rows = []
    for name, preds in baselines.items():
        r = rouge_service.evaluate(preds, refs)
        e = extra_service.evaluate(preds, refs, articles)
        rows.append({"baseline": name, "n": len(refs), "rouge1": r.rouge1 * 100, "rouge2": r.rouge2 * 100,
                     "rougeL": r.rougeL * 100, "chrf_pp": e.chrf_pp, "bertscore_f1": e.bertscore_f1 * 100,
                     "unsupported_entity_rate": e.unsupported_entity_rate * 100})
    table = pd.DataFrame(rows)
    table.to_csv(output, index=False, float_format="%.4f")
    print(table.round(2).to_string(index=False))
    print(f"[INFO] Written to {output}")


if __name__ == "__main__":
    main()
