"""Entry point: runs C1-C4 on the same held-out samples (default 200; --all-data: the whole test split) and reports ROUGE."""
import argparse
import os
import random

import numpy as np
import torch

from application.use_cases import (ExportResultsUseCase, RunPromptSummarizationUseCase,
                                   TrainLoraSummarizerUseCase)
from domain.entities import Condition
from domain.services.evaluation_service import EvaluationService
from domain.services.prompt_builder import PromptBuilder, split_few_shot_pool
from domain.services.summary_parser import SummaryParserService
from domain.services.supplementary_metrics import SupplementaryEvaluationService
from infrastructure.csv_exporter import CsvResultExporter
from infrastructure.dataset_repository import TurkishNewsDatasetRepository
from infrastructure.hf_model_adapter import DEFAULT_MODEL, HuggingFaceModelAdapter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("BASE_MODEL", DEFAULT_MODEL))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--use-validation", action="store_true",
                   help="Evaluate on the validation split every epoch and keep the epoch with the lowest validation loss")
    p.add_argument("--all-data", action="store_true",
                   help="Use every usable train row (minus 3 few-shot examples) and the whole test split")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--output-dir", default="outputs", help="Directory for all generated artifacts")
    p.add_argument("--adapter-dir", default=None, help="Default: <output-dir>/lora_adapter")
    p.add_argument("--skip-training", action="store_true", help="Reuse an existing adapter in --adapter-dir")
    p.add_argument("--no-supplementary", action="store_true", help="Skip chrF++/BERTScore/entity metrics")
    p.add_argument("--raw-output", default=None, help="Default: <output-dir>/raw_generations.csv")
    p.add_argument("--metrics-output", default=None, help="Default: <output-dir>/metrics.csv")
    p.add_argument("--output", default=None, help="Default: <output-dir>/predictions.csv")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.adapter_dir = args.adapter_dir or os.path.join(args.output_dir, "lora_adapter")
    args.raw_output = args.raw_output or os.path.join(args.output_dir, "raw_generations.csv")
    args.metrics_output = args.metrics_output or os.path.join(args.output_dir, "metrics.csv")
    args.output = args.output or os.path.join(args.output_dir, "predictions.csv")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"[INFO] Loading dataset (seed={args.seed})")
    n_few_shot = 3
    if args.all_data:
        pool, test = TurkishNewsDatasetRepository(args.seed).load_splits(None, None)
        args.n_train = len(pool) - n_few_shot
    else:
        pool, test = TurkishNewsDatasetRepository(args.seed).load_splits(args.n_train + n_few_shot, args.n_test)
    train, few_shot = split_few_shot_pool(pool, args.n_train, n_few_shot)
    print(f"[INFO] train={len(train)} few_shot={len(few_shot)} test={len(test)}")
    print(f"[INFO] Few-shot example ids (random, not in training set): {[a.id for a in few_shot]}")

    builder = PromptBuilder(few_shot_examples=few_shot)
    parser = SummaryParserService(max_sentences=2)

    print(f"[INFO] Loading {args.model} in 4-bit NF4")
    model = HuggingFaceModelAdapter(args.model, args.seed)
    runner = RunPromptSummarizationUseCase(model, builder, parser, args.max_new_tokens, args.batch_size)

    summaries = {}
    for cond in (Condition.C1_ZERO_SHOT, Condition.C2_FEW_SHOT, Condition.C3_COT):
        print(f"[INFO] Running {cond.value}")
        summaries[cond] = runner.execute(test, cond)

    if args.skip_training:
        model.load_adapter(args.adapter_dir)
    else:
        validation = TurkishNewsDatasetRepository(args.seed).load_validation(None) if args.use_validation else []
        TrainLoraSummarizerUseCase(model, builder).execute(train, args.adapter_dir, validation)
    print(f"[INFO] Running {Condition.C4_LORA.value}")
    summaries[Condition.C4_LORA] = runner.execute(test, Condition.C4_LORA)

    CsvResultExporter().export_raw(summaries, args.raw_output)
    supplementary = None if args.no_supplementary else SupplementaryEvaluationService()
    ExportResultsUseCase(EvaluationService(), CsvResultExporter(), supplementary).execute(
        test, summaries, args.output, args.metrics_output)


if __name__ == "__main__":
    main()
