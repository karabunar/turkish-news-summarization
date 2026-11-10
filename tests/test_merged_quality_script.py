import pytest

pytest.importorskip("pandas")

from scripts import merged_quality


def test_merged_quality_script_exposes_main():
    assert callable(merged_quality.main)
