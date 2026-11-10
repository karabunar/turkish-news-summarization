"""Helper of the C3 token-budget script."""
from scripts.cot_budget import marker_count


def test_marker_count_counts_raw_outputs_with_final_summary_marker_in_any_casing():
    raw = ["analiz\nNihai özet: kısa", "### Başlık\n1. madde", "NİHAİ ÖZET: x", "nihai özet"]
    assert marker_count(raw) == 3
