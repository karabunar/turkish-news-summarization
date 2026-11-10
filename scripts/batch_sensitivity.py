"""Does the batch composition change greedy C1 outputs? (checks the 'probably batching' explanation)

The same C1 prompts are generated for the 200 test articles that the 1000/200 run and the full run share:
  A1, A2  the 200 articles alone, batch size 8, run twice          -> is generation deterministic at all?
  B       all 273 test articles, batch size 8 (the main-run setting), 200 extracted -> reproduces the main run?
  C4      the 200 articles alone, batch size 4
  C1b     the 200 articles alone, batch size 1                     -> no padding at all
The raw outputs are compared pairwise (identical count) and with the stored C1 outputs of both earlier runs.
Reading: A1 = A2 and A1 != B means batch composition matters; A1 != A2 means GPU nondeterminism;
if A1 differs from the stored 1000/200 outputs although the batch setting is the same, the code changed in between.
Only C1 inference, no adapter. Needs the GPU (a few minutes).

Usage (project root, inside the container, GPU free):
    python -m scripts.batch_sensitivity --output-dir outputs/batch_sensitivity
"""
import argparse
import itertools
import os
from typing import Dict, List, Sequence

import pandas as pd


def identical_count(a: Sequence[str], b: Sequence[str]) -> int:
    return sum(x == y for x, y in zip(a, b))


def pairwise_table(outputs: Dict[str, Sequence[str]]) -> List[dict]:
    return [{"a": x, "b": y, "identical": identical_count(outputs[x], outputs[y]), "n": len(outputs[x])}
            for x, y in itertools.combinations(outputs, 2)]


def main() -> None:
    from domain.entities import Condition
    from domain.services.prompt_builder import PromptBuilder
    from domain.services.summary_parser import SummaryParserService
    from infrastructure.dataset_repository import TurkishNewsDatasetRepository
    from infrastructure.csv_exporter import read_predictions
    from infrastructure.hf_model_adapter import DEFAULT_MODEL, HuggingFaceModelAdapter
    from scripts.error_analysis import rouge_sample

    ap = argparse.ArgumentParser()
    ap.add_argument("--old-predictions", default="experiments/04_final_1000train_200test/predictions.csv")
    ap.add_argument("--old-raw", default="experiments/04_final_1000train_200test/raw_generations.csv")
    ap.add_argument("--new-predictions", default="outputs/predictions.csv")
    ap.add_argument("--new-raw", default="outputs/raw_generations.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--output-dir", default="outputs/batch_sensitivity")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _, test = TurkishNewsDatasetRepository(42).load_splits(1, None)
    old_pred = read_predictions(args.old_predictions).set_index("sample_id")
    shared = [a for a in test if a.id in old_pred.index]
    shared_ids = [a.id for a in shared]
    print(f"[INFO] test={len(test)} shared with the 1000/200 run={len(shared)}")

    builder, parser = PromptBuilder(), SummaryParserService(max_sentences=2)
    model = HuggingFaceModelAdapter(args.model, 42)

    def run(articles, batch_size):
        conv = [builder.build(Condition.C1_ZERO_SHOT, a.article_text) for a in articles]
        return dict(zip([a.id for a in articles], model.generate(conv, args.max_new_tokens, batch_size)))

    runs = {}
    for name, arts, bs in [("A1 (200, b8)", shared, 8), ("A2 (200, b8, repeat)", shared, 8),
                           ("B (273, b8)", test, 8), ("C (200, b4)", shared, 4), ("D (200, b1)", shared, 1)]:
        print(f"[INFO] {name}")
        out = run(arts, bs)
        runs[name] = [out[i] for i in shared_ids]
    old_raw = pd.read_csv(args.old_raw, keep_default_na=False).set_index("sample_id")
    new_raw = pd.read_csv(args.new_raw, keep_default_na=False).set_index("sample_id")
    runs["stored 1000/200 run"] = [old_raw.loc[i, "raw_C1_ZeroShot"] for i in shared_ids]
    runs["stored full run"] = [new_raw.loc[i, "raw_C1_ZeroShot"] for i in shared_ids]

    table = pd.DataFrame(pairwise_table(runs))
    refs = [a.reference_summary for a in shared]
    scores = pd.DataFrame([{"run": k, "rougeL": 100 * sum(rouge_sample(parser.parse(t), r)[2] for t, r in zip(v, refs)) / len(v)}
                           for k, v in runs.items()])
    table.to_csv(os.path.join(args.output_dir, "pairwise_identical_raw.csv"), index=False)
    scores.to_csv(os.path.join(args.output_dir, "rougeL_per_run.csv"), index=False, float_format="%.4f")
    pd.DataFrame({"sample_id": shared_ids, **runs}).to_csv(os.path.join(args.output_dir, "raw_c1_runs.csv"), index=False)
    print(table.to_string(index=False))
    print(scores.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
