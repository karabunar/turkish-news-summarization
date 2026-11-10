"""Domain service that builds chat messages for conditions C1-C4."""
from typing import Dict, List, Sequence

from domain.entities import Condition, NewsArticle

Message = Dict[str, str]

ZERO_SHOT_INSTRUCTION = "Aşağıdaki haber metnini 1-2 cümle ile anlaşılır şekilde özetle:"
COT_PERSONA = (
    "Sen uzman bir haber editörüsün. Haber metnindeki ana olayı, kişileri ve sonucu "
    "adım adım analiz et, ardından nihai özeti yaz:"
)
COT_FORMAT_HINT = (
    "Analizi kısa tut (en fazla 3 kısa madde). Cevabın son satırı 'Nihai özet:' ile "
    "başlamalı ve 1-2 cümle içermelidir."
)
FINAL_SUMMARY_MARKER = "Nihai özet:"


def split_few_shot_pool(pool: Sequence[NewsArticle], n_train: int, n_few_shot: int = 3):
    """Split a seeded random pool into (training articles, few-shot examples); the two never overlap."""
    if len(pool) != n_train + n_few_shot:
        raise ValueError(f"Pool must hold n_train + n_few_shot = {n_train + n_few_shot} articles.")
    return list(pool[:n_train]), list(pool[n_train:])


class PromptBuilder:
    """Builds chat-format prompts. C4 deliberately reuses the C1 prompt."""

    def __init__(self, few_shot_examples: Sequence[NewsArticle] = (),
                 max_article_chars: int = 2500, max_example_chars: int = 800):
        self._examples = list(few_shot_examples)
        self._max_article_chars = max_article_chars
        self._max_example_chars = max_example_chars

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        cut = text[:limit]
        if text[limit].isspace() or " " not in cut:
            return cut  # cut falls on a word boundary, or there is no space to back off to
        return cut.rsplit(" ", 1)[0]

    def _user_turn(self, instruction: str, text: str, limit: int) -> str:
        return f"{instruction}\n\n{self._clip(text, limit)}"

    def build(self, condition: Condition, article_text: str) -> List[Message]:
        if condition in (Condition.C1_ZERO_SHOT, Condition.C4_LORA):
            return [{"role": "user", "content": self._user_turn(
                ZERO_SHOT_INSTRUCTION, article_text, self._max_article_chars)}]
        if condition == Condition.C2_FEW_SHOT:
            if len(self._examples) != 3:
                raise ValueError("Few-shot condition requires exactly 3 examples.")
            messages: List[Message] = []
            for ex in self._examples:
                messages.append({"role": "user", "content": self._user_turn(
                    ZERO_SHOT_INSTRUCTION, ex.article_text, self._max_example_chars)})
                messages.append({"role": "assistant", "content": ex.reference_summary.strip()})
            messages.append({"role": "user", "content": self._user_turn(
                ZERO_SHOT_INSTRUCTION, article_text, self._max_article_chars)})
            return messages
        if condition == Condition.C3_COT:
            return [{"role": "user", "content": self._user_turn(
                f"{COT_PERSONA}\n{COT_FORMAT_HINT}", article_text, self._max_article_chars)}]
        raise ValueError(f"Unknown condition: {condition}")

    def build_training_conversation(self, article: NewsArticle) -> List[Message]:
        """C1-style prompt plus the reference summary as the assistant turn."""
        messages = self.build(Condition.C4_LORA, article.article_text)
        messages.append({"role": "assistant", "content": article.reference_summary.strip()})
        return messages
