"""Train/test leakage check (run from the project root: python -m scripts.check_leakage): exact and near-duplicate articles and summaries between the splits."""
import argparse
import re
from typing import Dict, List, Sequence, Set, Tuple


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.replace("I", "ı").replace("İ", "i").lower()).strip()


def shingles(text: str, n: int = 5) -> Set[Tuple[str, ...]]:
    words = normalize(text).split()
    return {tuple(words[i:i + n]) for i in range(max(len(words) - n + 1, 1))}


def jaccard(a: Set, b: Set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def find_overlaps(train: Sequence[str], test: Sequence[str], threshold: float = 0.5
                  ) -> Dict[str, List[Tuple[int, int, float]]]:
    """Return exact matches and near-duplicates (5-gram Jaccard >= threshold) as (test_idx, train_idx, score)."""
    train_norm = [normalize(t) for t in train]
    train_prefix: Dict[str, int] = {}
    for i, t in enumerate(train_norm):
        train_prefix.setdefault(t[:200], i)
    train_sh = [shingles(t) for t in train]
    exact, prefix, near = [], [], []
    for j, text in enumerate(test):
        norm = normalize(text)
        if not norm:  # empty texts would otherwise match each other perfectly
            continue
        if norm in train_norm:
            exact.append((j, train_norm.index(norm), 1.0))
            continue
        if norm[:200] in train_prefix:
            prefix.append((j, train_prefix[norm[:200]], 1.0))
        sh = shingles(text)
        score, best = max((jaccard(sh, train_sh[i]), i) for i in range(len(train)))
        if score >= threshold:
            near.append((j, best, score))
    return {"exact": exact, "same_first_200_chars": prefix, "near_duplicate": near}


def main() -> None:
    from infrastructure.dataset_repository import TurkishNewsDatasetRepository
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--show", type=int, default=3, help="Number of most similar flagged pairs to print")
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=200)
    args = p.parse_args()
    train, test = TurkishNewsDatasetRepository(42).load_splits(args.n_train, args.n_test)

    print(f"=== Article text: test ({len(test)}) vs train ({len(train)}) ===")
    art = find_overlaps([a.article_text for a in train], [a.article_text for a in test], args.threshold)
    for kind, hits in art.items():
        print(f"{kind}: {len(hits)}/{len(test)}")
    print("=== Reference summaries: test vs train ===")
    summ = find_overlaps([a.reference_summary for a in train], [a.reference_summary for a in test], 0.8)
    for kind, hits in summ.items():
        print(f"{kind}: {len(hits)}/{len(test)}")
    flagged = sorted({j for hits in art.values() for j, _, _ in hits})
    print("flagged test sample_ids (articles):", [test[j].id for j in flagged][:30])
    for j, i, s in sorted(art["near_duplicate"], key=lambda x: -x[2])[:args.show]:
        print(f"\n[{s:.2f}] TEST  {test[j].article_text[:150]!r}\n       TRAIN {train[i].article_text[:150]!r}")


if __name__ == "__main__":
    main()
