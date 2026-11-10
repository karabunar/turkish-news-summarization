"""Decoding and post-processing ablation for the fine-tuned model (C4), chosen on VALIDATION data only.

Base variants = decoding setting x parser limit (2 or 3 sentences). The base variant with the best validation
ROUGE-L is the one to report. Two optional post-processing steps are scored for every base variant but are NOT tuned:
`+repair` (replace unsupported names by the most similar article name, similarity >= 0.8) and `+faithful`
(beam variants only: of the 4 best beams take the one with the fewest unsupported entities/numbers).
The test split is scored for all variants; nothing is selected on it. Needs the GPU and a trained adapter.

Usage (project root, inside the container, after the main run):
    python -m scripts.decode_ablation --adapter-dir outputs/lora_adapter --output-dir outputs/decode_ablation
"""
import argparse
import os
from typing import Dict, List

import pandas as pd

DECODING: Dict[str, dict] = {
    "greedy-80": {"max_new_tokens": 80, "gen": {}, "batch": 8, "candidates": 1},         # the setting used for C4
    "greedy-120": {"max_new_tokens": 120, "gen": {}, "batch": 8, "candidates": 1},
    "beam4-80": {"max_new_tokens": 80, "gen": {"num_beams": 4, "early_stopping": True}, "batch": 4, "candidates": 4},
    "beam4-120": {"max_new_tokens": 120, "gen": {"num_beams": 4, "early_stopping": True}, "batch": 4, "candidates": 4},
}
MAX_SENTENCES = (2, 3)
REPAIR_SIMILARITY = 0.8


def select_best(rows: List[dict], key: str = "rougeL") -> str:
    """Name of the variant with the highest `key` among the given rows (ties: first listed)."""
    return max(rows, key=lambda r: r[key])["variant"]


def base_rows(rows: List[dict]) -> List[dict]:
    """Rows without post-processing: only these compete in the validation selection."""
    return [r for r in rows if "+" not in r["variant"]]


def main() -> None:
    from domain.services.entity_repair import pick_most_faithful, repair_entities
    from domain.services.evaluation_service import EvaluationService
    from domain.services.prompt_builder import PromptBuilder
    from domain.services.summary_parser import SummaryParserService
    from domain.services.supplementary_metrics import count_unsupported
    from domain.entities import Condition
    from infrastructure.dataset_repository import TurkishNewsDatasetRepository
    from infrastructure.hf_model_adapter import DEFAULT_MODEL, HuggingFaceModelAdapter

    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", default="outputs/lora_adapter")
    ap.add_argument("--output-dir", default="outputs/decode_ablation")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-test", type=int, default=10**9, help="Number of test samples (default: all)")
    ap.add_argument("--n-val", type=int, default=10**9, help="Number of validation samples (default: all)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    repo = TurkishNewsDatasetRepository(42)
    val = repo.load_validation(args.n_val)
    _, test = repo.load_splits(1, args.n_test)
    model = HuggingFaceModelAdapter(args.model, 42)
    model.load_adapter(args.adapter_dir)
    builder, evaluator = PromptBuilder(), EvaluationService()

    def score(split: str, name: str, texts: List[str], articles) -> dict:
        refs = [a.reference_summary for a in articles]
        r = evaluator.evaluate(texts, refs)
        unsupported = total = 0
        for t, a in zip(texts, articles):
            u, n = count_unsupported(t, a.article_text)
            unsupported, total = unsupported + u, total + n
        return {"split": split, "variant": name, "n": len(articles), "rouge1": r.rouge1 * 100, "rouge2": r.rouge2 * 100,
                "rougeL": r.rougeL * 100, "mean_words": sum(len(t.split()) for t in texts) / len(texts),
                "unsupported_pct": 100 * unsupported / total if total else 0.0}

    rows: List[dict] = []
    saved: Dict[str, Dict[str, List[str]]] = {}
    for split, articles in (("validation", val), ("test", test)):
        conversations = [builder.build(Condition.C4_LORA, a.article_text) for a in articles]
        for dname, cfg in DECODING.items():
            print(f"[INFO] {split}: {dname} ({len(articles)} samples)")
            raw = model.generate_candidates(conversations, cfg["max_new_tokens"], cfg["batch"],
                                            cfg["candidates"], cfg["gen"])
            for k in MAX_SENTENCES:
                parser = SummaryParserService(max_sentences=k)
                parsed = [[parser.parse(c) for c in cands] for cands in raw]
                top = [p[0] for p in parsed]
                name = f"{dname}/s{k}"
                rows.append(score(split, name, top, articles))
                saved.setdefault(name, {})[split] = top
                rows.append(score(split, name + "+repair",
                                  [repair_entities(t, a.article_text, REPAIR_SIMILARITY) for t, a in zip(top, articles)],
                                  articles))
                if cfg["candidates"] > 1:
                    rows.append(score(split, name + "+faithful",
                                      [pick_most_faithful(p, a.article_text) for p, a in zip(parsed, articles)], articles))
            pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "decode_ablation.csv"), index=False, float_format="%.4f")

    table = pd.DataFrame(rows)
    chosen = select_best(base_rows([r for r in rows if r["split"] == "validation"]))
    print(table.round(2).to_string(index=False))
    print(f"[INFO] Base variant selected on validation: {chosen}")
    pd.DataFrame({"sample_id": [a.id for a in test], "reference_summary": [a.reference_summary for a in test],
                  "pred_selected": saved[chosen]["test"]}).to_csv(
        os.path.join(args.output_dir, "predictions_selected_variant.csv"), index=False)


if __name__ == "__main__":
    main()
