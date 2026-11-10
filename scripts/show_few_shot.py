"""Print the few-shot examples that main.py will use (random, seeded, outside the training set).

Usage (project root): python -m scripts.show_few_shot [--n-train 4907]
"""
import argparse

from domain.services.prompt_builder import split_few_shot_pool
from infrastructure.dataset_repository import TurkishNewsDatasetRepository


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=1000)
    args = p.parse_args()
    pool, _ = TurkishNewsDatasetRepository(42).load_splits(args.n_train + 3, 1)
    _, few_shot = split_few_shot_pool(pool, args.n_train)
    for ex in few_shot:
        print(f"--- id={ex.id} | article chars={len(ex.article_text)}")
        print("ARTICLE :", ex.article_text[:300].replace("\n", " "))
        print("SUMMARY :", ex.reference_summary)


if __name__ == "__main__":
    main()
