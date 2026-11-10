"""Loads the Turkish news summarization dataset and maps rows to domain entities."""
from typing import List, Optional, Tuple

from datasets import load_dataset

from domain.entities import NewsArticle

DATASET_ID = "ahmetyaylalioglu/turkish-news-summarization-collection"


class TurkishNewsDatasetRepository:
    def __init__(self, seed: int = 42, dataset_id: str = DATASET_ID):
        self._seed = seed
        self._dataset_id = dataset_id

    @staticmethod
    def _to_entity(row: dict, fallback_id: str) -> NewsArticle:
        return NewsArticle(id=str(row["id"] if row.get("id") not in (None, "") else fallback_id),
                           article_text=row["text"], reference_summary=row["target_summary"])

    def _sample(self, split: str, n: Optional[int]) -> List[NewsArticle]:
        ds = load_dataset(self._dataset_id, split=split)
        ds = ds.filter(lambda r: bool(r["text"] and r["target_summary"]))
        ds = ds.shuffle(seed=self._seed).select(range(len(ds) if n is None else min(n, len(ds))))
        return [self._to_entity(r, f"{split}-{i}") for i, r in enumerate(ds)]

    def load_validation(self, n: Optional[int] = 273) -> List[NewsArticle]:
        """Samples from the `validation` split (used only to choose decoding settings)."""
        return self._sample("validation", n)

    def load_splits(self, n_train: Optional[int] = 1000, n_test: Optional[int] = 200) -> Tuple[List[NewsArticle], List[NewsArticle]]:
        """Train samples come from the `train` split, test samples from the `test` split (no overlap). None = whole split."""
        return self._sample("train", n_train), self._sample("test", n_test)
