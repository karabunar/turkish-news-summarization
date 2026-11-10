"""Few-shot (C2) sensitivity on the full test split: how much do the three in-context examples matter?

C2 is run with the example triple of the main run, with several other random triples from the training pool,
and with the three shortest training articles (the choice of the earlier 1000/200 run). Only inference, the
adapter is not needed. The test split is scored for every triple; nothing is selected on it.

Usage (project root, inside the container, GPU free):
    python -m scripts.fewshot_sensitivity --n-sets 5 --output-dir outputs/fewshot_sensitivity
"""
import argparse
import os
import random
from typing import List, Sequence, Set

import numpy as np
import pandas as pd

from domain.entities import NewsArticle

N_FEW_SHOT = 3


def sample_triples(pool: Sequence[NewsArticle], n_sets: int, exclude_ids: Set[str], seed: int) -> List[List[NewsArticle]]:
    """`n_sets` distinct random example triples from `pool`, never containing an excluded id."""
    rng = random.Random(seed)
    candidates = [a for a in pool if a.id not in exclude_ids]
    triples, seen = [], set()
    while len(triples) < n_sets:
        t = rng.sample(candidates, N_FEW_SHOT)
        key = tuple(sorted(a.id for a in t))
        if key not in seen:
            seen.add(key)
            triples.append(t)
    return triples


def shortest_triple(pool: Sequence[NewsArticle]) -> List[NewsArticle]:
    return sorted(pool, key=lambda a: len(a.article_text))[:N_FEW_SHOT]


def main() -> None:
    from application.use_cases import RunPromptSummarizationUseCase
    from domain.entities import Condition
    from domain.services.evaluation_service import EvaluationService
    from domain.services.prompt_builder import PromptBuilder, split_few_shot_pool
    from domain.services.summary_parser import SummaryParserService
    from domain.services.supplementary_metrics import count_unsupported
    from infrastructure.dataset_repository import TurkishNewsDatasetRepository
    from infrastructure.hf_model_adapter import DEFAULT_MODEL, HuggingFaceModelAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sets", type=int, default=5, help="number of extra random example triples")
    ap.add_argument("--seed", type=int, default=123, help="seed of the triple sampling (not the data seed)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--output-dir", default="outputs/fewshot_sensitivity")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pool, test = TurkishNewsDatasetRepository(42).load_splits(None, None)
    train, main_triple = split_few_shot_pool(pool, len(pool) - N_FEW_SHOT, N_FEW_SHOT)
    variants = {"main (main-run set)": main_triple, "3 shortest": shortest_triple(train)}
    for i, t in enumerate(sample_triples(train, args.n_sets, {a.id for a in main_triple}, args.seed), 1):
        variants[f"random {i}"] = t

    model = HuggingFaceModelAdapter(args.model, 42)
    evaluator, parser = EvaluationService(), SummaryParserService(max_sentences=2)
    refs = [a.reference_summary for a in test]
    rows, preds = [], {"sample_id": [a.id for a in test]}
    for name, triple in variants.items():
        print(f"[INFO] C2 with {name}: {[a.id for a in triple]}")
        runner = RunPromptSummarizationUseCase(model, PromptBuilder(few_shot_examples=triple), parser,
                                               args.max_new_tokens, args.batch_size)
        texts = [s.generated_text for s in runner.execute(test, Condition.C2_FEW_SHOT)]
        r = evaluator.evaluate(texts, refs)
        unsupported = total = 0
        for t, a in zip(texts, test):
            u, n = count_unsupported(t, a.article_text)
            unsupported, total = unsupported + u, total + n
        rows.append({"variant": name, "example_ids": " ".join(a.id for a in triple),
                     "example_chars": sum(len(a.article_text) for a in triple),
                     "rouge1": r.rouge1 * 100, "rouge2": r.rouge2 * 100, "rougeL": r.rougeL * 100,
                     "unsupported_pct": 100 * unsupported / total if total else 0.0})
        preds[name] = texts
        pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "fewshot_sensitivity.csv"), index=False, float_format="%.4f")
        pd.DataFrame(preds).to_csv(os.path.join(args.output_dir, "predictions_c2_variants.csv"), index=False)

    table = pd.DataFrame(rows)
    print(table.round(2).to_string(index=False))
    rl = table["rougeL"].to_numpy()
    rand = table[table["variant"].str.startswith("random")]["rougeL"].to_numpy()
    print(f"[INFO] random triples: mean R-L {rand.mean():.2f}, sd {rand.std(ddof=1):.2f}, "
          f"min {rand.min():.2f}, max {rand.max():.2f}; all variants range {np.ptp(rl):.2f}")


if __name__ == "__main__":
    main()
