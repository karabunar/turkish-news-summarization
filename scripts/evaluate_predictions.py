"""Recompute all metrics from an existing predictions CSV and write metrics.csv.

Usage (project root): python -m scripts.evaluate_predictions [--predictions outputs/predictions.csv]
With --ids-from <other predictions csv> only the samples (by sample_id) that also occur in that file are scored,
e.g. to compare a full-data run with an earlier run on exactly the same test articles.
"""
import argparse
import os

import pandas as pd

from domain.services.evaluation_service import EvaluationService
from domain.services.supplementary_metrics import SupplementaryEvaluationService
from application.use_cases import format_results_table, format_supplementary_table
from infrastructure.csv_exporter import COLUMN_NAMES, CsvResultExporter, read_predictions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", default="outputs/predictions.csv")
    p.add_argument("--output", default=None, help="Default: metrics.csv next to the predictions file")
    p.add_argument("--no-supplementary", action="store_true")
    p.add_argument("--ids-from", default=None, help="Restrict to the sample_ids found in this predictions CSV")
    args = p.parse_args()
    output = args.output or os.path.join(os.path.dirname(args.predictions), "metrics.csv")

    frame = read_predictions(args.predictions)
    if args.ids_from:
        frame = frame[frame["sample_id"].isin(set(pd.read_csv(args.ids_from)["sample_id"]))].reset_index(drop=True)
    references = frame["reference_summary"].tolist()
    articles = frame["news_text"].tolist()
    predictions = {c: frame[col].tolist() for c, col in COLUMN_NAMES.items()}

    rouge = {c: EvaluationService().evaluate(p_, references) for c, p_ in predictions.items()}
    extra = None
    if not args.no_supplementary:
        service = SupplementaryEvaluationService()
        extra = {c: service.evaluate(p_, references, articles) for c, p_ in predictions.items()}

    CsvResultExporter().export_metrics(rouge, extra, len(frame), output)
    print(f"[INFO] Metrics written to {output}")
    print(format_results_table(rouge))
    if extra:
        print(format_supplementary_table(extra))


if __name__ == "__main__":
    main()
