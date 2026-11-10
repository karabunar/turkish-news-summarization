"""Domain entities and value objects for the summarization experiment."""
from dataclasses import dataclass
from enum import Enum


class Condition(str, Enum):
    """Experimental conditions: zero-shot, few-shot, persona+CoT prompting and QLoRA fine-tuning."""

    C1_ZERO_SHOT = "C1_ZeroShot"
    C2_FEW_SHOT = "C2_FewShot"
    C3_COT = "C3_CoT"
    C4_LORA = "C4_LoRA"


@dataclass(frozen=True)
class NewsArticle:
    id: str
    article_text: str
    reference_summary: str


@dataclass(frozen=True)
class GeneratedSummary:
    sample_id: str
    condition: Condition
    generated_text: str
    raw_text: str = ""


@dataclass(frozen=True)
class RougeMetricResult:
    rouge1: float
    rouge2: float
    rougeL: float
    rougeLsum: float


@dataclass(frozen=True)
class SupplementaryMetricResult:
    """chrF++ (0-100), BERTScore F1 (0-1) and corpus-level unsupported entity/number rate (0-1)."""

    chrf_pp: float
    bertscore_f1: float
    unsupported_entity_rate: float
