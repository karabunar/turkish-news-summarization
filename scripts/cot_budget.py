"""C3 (persona + chain-of-thought) with a larger token budget: is the weak C3 result a budget artefact?

With 80 new tokens many raw outputs end inside the analysis, before the final line 'Nihai özet:', and the parser then
falls back to another line (for Qwen2.5-3B typically a markdown heading). This script runs only C3 with larger budgets
and reports ROUGE, chrF++, BERTScore, the unsupported rate and how many raw outputs reach the marker. It is an ablation:
the main tables keep 80 tokens for every condition. No adapter is needed; the test split is scored, nothing is selected.

Usage (project root, inside the container, GPU free):
    python -m scripts.cot_budget --model Qwen/Qwen2.5-3B-Instruct --output-dir experiments/05_3b_full_data/cot_budget
    python -m scripts.cot_budget --output-dir outputs/cot_budget          # the 1.5B model
"""
import argparse
import os
import re
from typing import Sequence

import pandas as pd

MARKER = re.compile(r"n[iİI]ha[iİI]\s+[öo]zet", re.IGNORECASE)


def marker_count(raw_outputs: Sequence[str]) -> int:
    """Number of raw outputs that contain the final-summary marker."""
    return sum(bool(MARKER.search(t)) for t in raw_outputs)


def main() -> None:
    from application.use_cases import RunPromptSummarizationUseCase
    from domain.entities import Condition
    from domain.services.evaluation_service import EvaluationService
    from domain.services.prompt_builder import PromptBuilder
    from domain.services.summary_parser import SummaryParserService
    from domain.services.supplementary_metrics import SupplementaryEvaluationService
    from infrastructure.dataset_repository import TurkishNewsDatasetRepository
    from infrastructure.hf_model_adapter import DEFAULT_MODEL, HuggingFaceModelAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--budgets", type=int, nargs="+", default=[160], help="new-token budgets to try (80 is the main setting)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output-dir", default="outputs/cot_budget")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _, test = TurkishNewsDatasetRepository(42).load_splits(1, None)
    refs, arts = [a.reference_summary for a in test], [a.article_text for a in test]
    model = HuggingFaceModelAdapter(args.model, 42)
    parser, evaluator, extra = SummaryParserService(max_sentences=2), EvaluationService(), SupplementaryEvaluationService()

    rows, preds, raws = [], {"sample_id": [a.id for a in test]}, {"sample_id": [a.id for a in test]}
    for budget in args.budgets:
        print(f"[INFO] C3 with max_new_tokens={budget} on {len(test)} test articles")
        runner = RunPromptSummarizationUseCase(model, PromptBuilder(), parser, budget, args.batch_size)
        out = runner.execute(test, Condition.C3_COT)
        texts, raw = [s.generated_text for s in out], [s.raw_text for s in out]
        r, e = evaluator.evaluate(texts, refs), extra.evaluate(texts, refs, arts)
        rows.append({"model": args.model, "max_new_tokens": budget, "n": len(test),
                     "rouge1": r.rouge1 * 100, "rouge2": r.rouge2 * 100, "rougeL": r.rougeL * 100,
                     "chrf_pp": e.chrf_pp, "bertscore_f1": e.bertscore_f1 * 100,
                     "unsupported_pct": e.unsupported_entity_rate * 100,
                     "with_marker": marker_count(raw), "mean_words": sum(len(t.split()) for t in texts) / len(texts)})
        preds[f"pred_{budget}"], raws[f"raw_{budget}"] = texts, raw
        pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "cot_budget.csv"), index=False, float_format="%.4f")
        pd.DataFrame(preds).to_csv(os.path.join(args.output_dir, "predictions_c3.csv"), index=False)
        pd.DataFrame(raws).to_csv(os.path.join(args.output_dir, "raw_c3.csv"), index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
