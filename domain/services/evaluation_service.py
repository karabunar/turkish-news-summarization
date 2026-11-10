"""ROUGE evaluation for Turkish summaries via Hugging Face `evaluate`."""
import re
from typing import Sequence

from domain.entities import RougeMetricResult


class TurkishTokenizer:
    """Unicode-aware tokenizer; the default rouge_score tokenizer drops ç,ğ,ı,ö,ş,ü."""

    def __call__(self, text: str):
        # `evaluate` wraps the tokenizer and calls it as a function.
        return self.tokenize(text)

    def tokenize(self, text: str):
        text = text.replace("I", "ı").replace("İ", "i").lower()
        return re.findall(r"\w+", text, flags=re.UNICODE)


class EvaluationService:
    def __init__(self, rouge_metric=None):
        if rouge_metric is None:
            import evaluate
            rouge_metric = evaluate.load("rouge")
        self._rouge = rouge_metric

    def evaluate(self, predictions: Sequence[str], references: Sequence[str]) -> RougeMetricResult:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length.")
        if not predictions:
            raise ValueError("Cannot evaluate an empty prediction list.")
        scores = self._rouge.compute(
            predictions=list(predictions), references=list(references),
            tokenizer=TurkishTokenizer(), use_aggregator=False, use_stemmer=False,
        )
        # The default aggregator bootstraps, so scores change slightly between runs; plain means do not.
        mean = lambda key: sum(scores[key]) / len(scores[key])
        return RougeMetricResult(
            rouge1=mean("rouge1"), rouge2=mean("rouge2"),
            rougeL=mean("rougeL"), rougeLsum=mean("rougeLsum"),
        )
