"""Post-processing that targets unsupported (invented or garbled) names in generated summaries."""
import re
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Set

from domain.services.evaluation_service import TurkishTokenizer
from domain.services.supplementary_metrics import MIN_PREFIX_LEN, is_supported, count_unsupported

_TOKEN = re.compile(r"\w+(?:['’]\w+)?", re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_APOSTROPHE = re.compile(r"['’]")
_TOKENIZER = TurkishTokenizer()


def _key(word: str) -> str:
    tokens = _TOKENIZER.tokenize(word)
    return tokens[0] if tokens else word.lower()


def _article_names(article: str) -> Dict[str, str]:
    """Capitalized article words (suffix after an apostrophe removed), keyed by normalized form."""
    names: Dict[str, str] = {}
    for token in _TOKEN.findall(article):
        stem = _APOSTROPHE.split(token)[0]
        if stem[:1].isupper() and not stem.isdigit() and len(stem) >= MIN_PREFIX_LEN:
            names.setdefault(_key(stem), stem)
    return names


def repair_entities(text: str, article: str, min_similarity: float = 0.8) -> str:
    """Replace capitalized words that do not occur in the article by the most similar capitalized article word.

    Only non-sentence-initial words of at least four letters are touched, digits never are, and a replacement
    needs a character similarity (difflib ratio on normalized forms) of at least `min_similarity`.
    The suffix after an apostrophe is kept.
    """
    article_tokens: Set[str] = set(_TOKENIZER.tokenize(article))
    names = _article_names(article)

    def repair_sentence(sentence: str) -> str:
        position = -1

        def fix(match: "re.Match") -> str:
            nonlocal position
            position += 1
            token = match.group(0)
            parts = _APOSTROPHE.split(token, maxsplit=1)
            stem = parts[0]
            if (position == 0 or not stem[:1].isupper() or stem.isdigit() or len(stem) < MIN_PREFIX_LEN
                    or is_supported(_key(stem), article_tokens)):
                return token
            best, best_ratio = None, 0.0
            for normalized, original in names.items():
                ratio = SequenceMatcher(None, _key(stem), normalized).ratio()
                if ratio > best_ratio:
                    best, best_ratio = original, ratio
            if best is None or best_ratio < min_similarity:
                return token
            return best + token[len(stem):]

        return _TOKEN.sub(fix, sentence)

    return " ".join(repair_sentence(s) for s in _SENTENCE_SPLIT.split(text)) if text else text


def pick_most_faithful(candidates: Sequence[str], article: str) -> str:
    """Candidate with the fewest unsupported entities/numbers; ties go to the earliest (best-ranked) candidate."""
    if not candidates:
        return ""
    counts: List[int] = [count_unsupported(c, article)[0] for c in candidates]
    return candidates[counts.index(min(counts))]
