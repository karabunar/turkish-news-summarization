"""Complementary metrics: chrF++, BERTScore (Turkish BERT) and unsupported-entity rate."""
import re
from typing import List, Sequence, Set

from domain.entities import SupplementaryMetricResult
from domain.services.evaluation_service import TurkishTokenizer

DEFAULT_BERT_MODEL = "dbmdz/bert-base-turkish-cased"
_TOKEN = re.compile(r"\w+(?:['’]\w+)?", re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
MIN_PREFIX_LEN = 4


def extract_entities_and_numbers(text: str) -> List[str]:
    """Capitalized non-sentence-initial words (suffix after an apostrophe removed) and digit groups."""
    items: List[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        tokens = _TOKEN.findall(sentence)
        for position, token in enumerate(tokens):
            stem = re.split(r"['’]", token)[0]
            if stem.isdigit():
                items.append(stem)
            elif position > 0 and stem[:1].isupper():
                items.append(stem)
    return items


def is_supported(key: str, article_tokens: Set[str]) -> bool:
    """Exact token match; long words also match inflected forms (prefix). Digits and short words never do."""
    if key in article_tokens:
        return True
    if key.isdigit() or len(key) < MIN_PREFIX_LEN:
        return False
    return any(tok.startswith(key) for tok in article_tokens)


def count_unsupported(prediction: str, article: str) -> "tuple[int, int]":
    """Return (unsupported, total) entities/numbers of `prediction` absent from `article`."""
    tokenizer = TurkishTokenizer()
    article_tokens: Set[str] = set(tokenizer.tokenize(article))
    items = extract_entities_and_numbers(prediction)
    unsupported = 0
    for item in items:
        key = tokenizer.tokenize(item)[0] if tokenizer.tokenize(item) else item
        if not is_supported(key, article_tokens):
            unsupported += 1
    return unsupported, len(items)


class SupplementaryEvaluationService:
    def __init__(self, chrf_metric=None, bertscore_metric=None,
                 bert_model: str = DEFAULT_BERT_MODEL, bert_num_layers: int = 9):
        if chrf_metric is None or bertscore_metric is None:
            import evaluate
            chrf_metric = chrf_metric or evaluate.load("chrf")
            bertscore_metric = bertscore_metric or evaluate.load("bertscore")
        self._chrf = chrf_metric
        self._bertscore = bertscore_metric
        self._bert_model = bert_model
        self._bert_num_layers = bert_num_layers

    def evaluate(self, predictions: Sequence[str], references: Sequence[str],
                 articles: Sequence[str]) -> SupplementaryMetricResult:
        if not (len(predictions) == len(references) == len(articles)):
            raise ValueError("predictions, references and articles must have the same length.")
        if not predictions:
            raise ValueError("Cannot evaluate an empty prediction list.")
        chrf = self._chrf.compute(predictions=list(predictions),
                                  references=[[r] for r in references], word_order=2)
        bert = self._bertscore.compute(predictions=list(predictions), references=list(references),
                                       lang="tr", model_type=self._bert_model,
                                       num_layers=self._bert_num_layers)
        f1 = sum(bert["f1"]) / len(bert["f1"])
        unsupported = total = 0
        for pred, article in zip(predictions, articles):
            u, t = count_unsupported(pred, article)
            unsupported, total = unsupported + u, total + t
        rate = unsupported / total if total else 0.0
        return SupplementaryMetricResult(float(chrf["score"]), float(f1), rate)
