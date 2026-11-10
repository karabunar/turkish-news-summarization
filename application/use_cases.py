"""Application use cases orchestrating domain services and infrastructure ports."""
from typing import Dict, List, Optional, Sequence

from application.ports import LoraTrainer, ResultExporter, TextGenerator
from domain.entities import (Condition, GeneratedSummary, NewsArticle, RougeMetricResult,
                             SupplementaryMetricResult)
from domain.services.evaluation_service import EvaluationService
from domain.services.prompt_builder import PromptBuilder
from domain.services.summary_parser import SummaryParserService
from domain.services.supplementary_metrics import SupplementaryEvaluationService


class RunPromptSummarizationUseCase:
    """Runs summarization for one condition (C1, C2, C3 or C4) over the test set."""

    def __init__(self, generator: TextGenerator, prompt_builder: PromptBuilder,
                 parser: SummaryParserService, max_new_tokens: int = 80, batch_size: int = 8):
        self._generator = generator
        self._builder = prompt_builder
        self._parser = parser
        self._max_new_tokens = max_new_tokens
        self._batch_size = batch_size

    def execute(self, articles: Sequence[NewsArticle], condition: Condition) -> List[GeneratedSummary]:
        conversations = [self._builder.build(condition, a.article_text) for a in articles]
        raw = self._generator.generate(conversations, self._max_new_tokens, self._batch_size)
        return [GeneratedSummary(a.id, condition, self._parser.parse(r, chain_of_thought=condition == Condition.C3_COT),
                                  raw_text=r)
                for a, r in zip(articles, raw)]


class TrainLoraSummarizerUseCase:
    """Runs 4-bit QLoRA fine-tuning for condition C4."""

    def __init__(self, trainer: LoraTrainer, prompt_builder: PromptBuilder):
        self._trainer = trainer
        self._builder = prompt_builder

    def execute(self, train_articles: Sequence[NewsArticle], output_dir: str,
                validation_articles: Sequence[NewsArticle] = ()) -> str:
        """With validation articles the trainer evaluates every epoch and keeps the epoch with the lowest validation loss."""
        conversations = [self._builder.build_training_conversation(a) for a in train_articles]
        eval_conversations = [self._builder.build_training_conversation(a) for a in validation_articles]
        return self._trainer.fine_tune(conversations, output_dir, eval_conversations)


def format_results_table(results: Dict[Condition, RougeMetricResult]) -> str:
    """Markdown table with ROUGE scores in percent."""
    lines = ["| Condition       | ROUGE-1 | ROUGE-2 | ROUGE-L |",
             "|-----------------|---------|---------|---------|"]
    for cond, r in results.items():
        lines.append(f"| {cond.value:<15} | {r.rouge1*100:7.2f} | {r.rouge2*100:7.2f} | {r.rougeL*100:7.2f} |")
    return "\n".join(lines)


def format_supplementary_table(results: Dict[Condition, SupplementaryMetricResult]) -> str:
    """Markdown table with chrF++, BERTScore F1 (percent) and unsupported entity rate (percent)."""
    lines = ["| Condition       | chrF++  | BERTScore-F1 | Unsupported Ent./Num. % |",
             "|-----------------|---------|--------------|-------------------------|"]
    for cond, r in results.items():
        lines.append(f"| {cond.value:<15} | {r.chrf_pp:7.2f} | {r.bertscore_f1*100:12.2f} "
                     f"| {r.unsupported_entity_rate*100:23.2f} |")
    return "\n".join(lines)


class ExportResultsUseCase:
    """Evaluates all conditions, exports the CSV and prints the summary table."""

    def __init__(self, evaluator: EvaluationService, exporter: ResultExporter,
                 supplementary: Optional[SupplementaryEvaluationService] = None):
        self._evaluator = evaluator
        self._exporter = exporter
        self._supplementary = supplementary

    def execute(self, articles: Sequence[NewsArticle],
                summaries: Dict[Condition, List[GeneratedSummary]], csv_path: str,
                metrics_path: Optional[str] = None) -> Dict[Condition, RougeMetricResult]:
        predictions = {c: [s.generated_text for s in items] for c, items in summaries.items()}
        references = [a.reference_summary for a in articles]
        results = {c: self._evaluator.evaluate(p, references) for c, p in predictions.items()}
        extra = None
        if self._supplementary is not None:
            texts = [a.article_text for a in articles]
            extra = {c: self._supplementary.evaluate(p, references, texts) for c, p in predictions.items()}
        self._exporter.export(articles, predictions, csv_path)
        print(f"[INFO] Predictions exported to {csv_path}")
        if metrics_path:
            self._exporter.export_metrics(results, extra, len(articles), metrics_path)
            print(f"[INFO] Metrics exported to {metrics_path}")
        print("\nROUGE evaluation metrics summary")
        print(format_results_table(results))
        if extra is not None:
            print("\nSupplementary metrics (chrF++, BERTScore with Turkish BERT, unsupported entities/numbers):")
            print(format_supplementary_table(extra))
        print("=" * 100)
        return results
