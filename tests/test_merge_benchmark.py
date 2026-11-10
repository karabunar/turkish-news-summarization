import math

from scripts.merge_benchmark import decode_ms_per_token, share_identical


def test_decode_time_excludes_prefill_and_first_token():
    # 1.0 s total, 0.1 s prefill (includes the first token), 10 tokens -> 0.9 s over 9 steps = 100 ms
    assert math.isclose(decode_ms_per_token(1.0, 0.1, 10), 100.0)


def test_decode_time_single_token_does_not_divide_by_zero():
    assert math.isclose(decode_ms_per_token(0.5, 0.2, 1), 300.0)


def test_share_identical_counts_equal_generations():
    assert share_identical([[1, 2], [3], [4]], [[1, 2], [9], [4]]) == 2 / 3


def test_share_identical_all_and_none():
    assert share_identical(["a", "b"], ["a", "b"]) == 1.0
    assert share_identical(["a", "b"], ["c", "d"]) == 0.0
