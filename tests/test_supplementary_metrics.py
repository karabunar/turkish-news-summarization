import pytest

from domain.services.supplementary_metrics import (SupplementaryEvaluationService, count_unsupported,
                                                   extract_entities_and_numbers)


class FakeChrf:
    def compute(self, **kw):
        assert kw["word_order"] == 2 and kw["references"] == [["r"]]
        return {"score": 42.0}


class FakeBert:
    def compute(self, **kw):
        assert kw["lang"] == "tr"
        return {"f1": [0.5, 0.7]}


def test_extracts_non_initial_capitalized_words_and_numbers():
    items = extract_entities_and_numbers("Bugün Erdoğan'ın açıklaması Ankara'da 25 kişi katıldı.")
    assert items == ["Erdoğan", "Ankara", "25"]


def test_suffixed_entity_is_supported_by_article():
    assert count_unsupported("Dün Ankara'da toplantı oldu.", "Toplantı Ankara'nın merkezinde yapıldı.") == (0, 1)


def test_hallucinated_entity_and_number_are_flagged():
    assert count_unsupported("Dün İzmir'de 90 kişi geldi.", "Ankara'da 25 kişi geldi.") == (2, 2)


def test_short_entity_needs_exact_token_not_prefix():
    # "Ali" must not be supported by "alışveriş"; "20" must not be supported by "2024".
    assert count_unsupported("Dün Ali geldi.", "Alışveriş merkezi açıldı.") == (1, 1)
    assert count_unsupported("Dün 20 kişi geldi.", "2024 yılında kişi geldi.") == (1, 1)


def test_long_entity_still_matches_suffixed_article_form():
    assert count_unsupported("Dün Ankara geldi.", "Ankaralılar toplandı.") == (0, 1)


def test_no_entities_gives_zero_total():
    assert count_unsupported("kısa bir özet", "metin") == (0, 0)


def test_service_aggregates_metrics():
    r = SupplementaryEvaluationService(FakeChrf(), FakeBert()).evaluate(
        ["Dün İzmir'de 90 kişi geldi."], ["r"], ["Ankara'da 25 kişi geldi."])
    assert r.chrf_pp == 42.0 and r.bertscore_f1 == pytest.approx(0.6)
    assert r.unsupported_entity_rate == 1.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        SupplementaryEvaluationService(FakeChrf(), FakeBert()).evaluate(["a"], ["r"], [])
