"""Example-set selection for the few-shot sensitivity script."""
from domain.entities import NewsArticle
from scripts.fewshot_sensitivity import shortest_triple, sample_triples


def _pool(n=50):
    return [NewsArticle(f"id{i}", "x" * (100 + (i * 7) % 40), f"s{i}") for i in range(n)]


def test_sample_triples_are_reproducible_distinct_and_exclude_given_ids():
    pool = _pool()
    a = sample_triples(pool, n_sets=4, exclude_ids={"id0", "id1"}, seed=7)
    b = sample_triples(pool, n_sets=4, exclude_ids={"id0", "id1"}, seed=7)
    assert [[x.id for x in t] for t in a] == [[x.id for x in t] for t in b]
    assert len({tuple(x.id for x in t) for t in a}) == 4
    assert all(len(t) == 3 and not {"id0", "id1"} & {x.id for x in t} for t in a)


def test_shortest_triple_returns_the_three_shortest_articles():
    pool = _pool()
    lengths = sorted(len(a.article_text) for a in pool)[:3]
    assert sorted(len(a.article_text) for a in shortest_triple(pool)) == lengths
