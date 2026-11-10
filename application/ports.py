"""Ports (interfaces) the application layer depends on."""
from typing import Dict, List, Optional, Protocol, Sequence

from domain.entities import Condition, NewsArticle

Message = Dict[str, str]


class TextGenerator(Protocol):
    def generate(self, conversations: Sequence[List[Message]], max_new_tokens: int = 80,
                 batch_size: int = 8, gen_kwargs: Optional[Dict] = None) -> List[str]: ...

    def generate_candidates(self, conversations: Sequence[List[Message]], max_new_tokens: int, batch_size: int,
                            num_candidates: int, gen_kwargs: Optional[Dict] = None) -> List[List[str]]: ...


class LoraTrainer(Protocol):
    def fine_tune(self, conversations: Sequence[List[Message]], output_dir: str,
                  eval_conversations: Sequence[List[Message]] = ()) -> str: ...


class ResultExporter(Protocol):
    def export(self, articles: Sequence[NewsArticle],
               predictions: Dict[Condition, List[str]], path: str) -> None: ...

    def export_raw(self, summaries: Dict[Condition, list], path: str) -> None: ...

    def export_metrics(self, rouge, supplementary, n_samples: int, path: str) -> None: ...
