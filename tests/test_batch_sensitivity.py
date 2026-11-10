"""Pairwise comparison helper of the batch-sensitivity script."""
from scripts.batch_sensitivity import identical_count, pairwise_table


def test_identical_count_counts_equal_positions():
    assert identical_count(["a", "b", "c"], ["a", "x", "c"]) == 2


def test_pairwise_table_has_one_row_per_pair_with_count_and_total():
    rows = pairwise_table({"p": ["a", "b"], "q": ["a", "c"], "r": ["a", "b"]})
    assert {(r["a"], r["b"]): (r["identical"], r["n"]) for r in rows} == {
        ("p", "q"): (1, 2), ("p", "r"): (2, 2), ("q", "r"): (1, 2)}
