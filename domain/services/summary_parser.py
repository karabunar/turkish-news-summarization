"""Pure domain service that cleans raw model outputs."""
import re
from typing import List

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")
_INITIALS = re.compile(r"^(?:[A-ZÇĞİÖŞÜ]\.)+$")
_ABBREVIATIONS = {"dr.", "prof.", "doç.", "av.", "sn.", "st.", "bkz.", "vb.", "vs.", "yy.", "mah.", "cad.", "sok.", "no."}
_FINAL_MARKER = re.compile(r"n[iİI]ha[iİI]\s+özet[^:\n]*:", re.IGNORECASE)
_LEADING_LABEL = re.compile(r"^\s*(özet|haber\s+özeti|summary)\s*:\s*", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries, but not after initials ("M.A.") or common abbreviations ("Dr.")."""
    sentences: List[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        last_word = text[start:match.start()].split()[-1] if text[start:match.start()].split() else ""
        if _INITIALS.match(last_word) or last_word.lower() in _ABBREVIATIONS:
            continue
        sentences.append(text[start:match.start()])
        start = match.end()
    sentences.append(text[start:])
    return [s for s in sentences if s.strip()]


class SummaryParserService:
    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    @staticmethod
    def _normalize(sentence: str) -> str:
        return re.sub(r"\W+", " ", sentence.lower()).strip()

    def parse(self, raw_output: str, chain_of_thought: bool = False) -> str:
        """Strip templates/labels, drop repeated sentences, truncate.

        `chain_of_thought` enables the C3-specific handling (final-summary marker, analysis bullets).
        """
        text = (raw_output or "").strip()
        # Chain-of-thought: keep only what follows the last final-summary marker.
        markers = list(_FINAL_MARKER.finditer(text)) if chain_of_thought else []
        if markers:
            text = text[markers[-1].end():]
        elif chain_of_thought:
            # No marker (e.g. generation cut off): drop analysis bullets/lines.
            lines = [ln for ln in text.splitlines() if ln.strip() and not _BULLET.match(ln)]
            text = " ".join(lines) if lines else text
        text = _LEADING_LABEL.sub("", " ".join(text.replace("*", "").split()))
        text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("*", "")
        text = " ".join(text.split())

        sentences: List[str] = split_sentences(text)
        unique, seen = [], set()
        for sentence in sentences:
            key = self._normalize(sentence)
            if key and key not in seen:
                seen.add(key)
                unique.append(sentence.strip())
        return " ".join(unique[: self.max_sentences])
