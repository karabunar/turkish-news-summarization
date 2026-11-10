import pytest

from domain.services.evaluation_service import EvaluationService, TurkishTokenizer


class FakeRouge:
    def compute(self, **kwargs):
        assert kwargs["tokenizer"]("ÇAĞRI") == ["çağrı"]
        assert kwargs["use_aggregator"] is False  # bootstrap aggregation is non-deterministic
        return {"rouge1": [0.25, 0.75], "rouge2": [0.0, 0.5], "rougeL": [0.2, 0.6], "rougeLsum": [0.2, 0.6]}


def test_turkish_tokenizer_keeps_turkish_characters():
    assert TurkishTokenizer().tokenize("Işık, İstanbul'da ğüşöç!") == ["ışık", "istanbul", "da", "ğüşöç"]


def test_averages_per_sample_scores_deterministically():
    result = EvaluationService(FakeRouge()).evaluate(["a"], ["b"])
    assert (result.rouge1, result.rouge2, result.rougeL) == (0.5, 0.25, 0.4)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        EvaluationService(FakeRouge()).evaluate(["a"], ["a", "b"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        EvaluationService(FakeRouge()).evaluate([], [])


def test_real_rouge_identical_text_scores_one_and_disjoint_zero():
    evaluate = pytest.importorskip("evaluate")
    try:
        svc = EvaluationService(evaluate.load("rouge"))
    except Exception:
        pytest.skip("rouge metric unavailable offline")
    same = svc.evaluate(["Çocuklar şehre gitti."], ["Çocuklar şehre gitti."])
    assert same.rouge1 == pytest.approx(1.0) and same.rougeL == pytest.approx(1.0)
    diff = svc.evaluate(["ağaç dal"], ["şehir yol"])
    assert diff.rouge1 == 0.0
