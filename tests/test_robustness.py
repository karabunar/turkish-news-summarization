import pytest

from scripts.check_leakage import find_overlaps


def test_falsy_id_is_kept_but_missing_id_falls_back():
    pytest.importorskip("datasets")
    from infrastructure.dataset_repository import TurkishNewsDatasetRepository
    row = {"text": "t", "target_summary": "s"}
    assert TurkishNewsDatasetRepository._to_entity({**row, "id": 0}, "x").id == "0"
    assert TurkishNewsDatasetRepository._to_entity({**row, "id": None}, "x").id == "x"
    assert TurkishNewsDatasetRepository._to_entity({**row, "id": ""}, "x").id == "x"


def test_leakage_ignores_empty_test_texts():
    result = find_overlaps(["", "bir iki üç dört beş altı yedi"], ["", "   "])
    assert all(not hits for hits in result.values())


def test_leakage_still_flags_near_duplicates():
    text = "bir iki üç dört beş altı yedi sekiz dokuz on"
    result = find_overlaps(["alakasız başka bir metin burada yer alıyor şimdi", text], [text + " on bir"])
    assert result["near_duplicate"] and result["near_duplicate"][0][1] == 1


def test_compare_runs_aligns_on_shared_sample_ids():
    import pandas as pd
    from scripts.compare_runs import align
    old = pd.DataFrame({"sample_id": ["a", "b"], "news_text": ["x", "y"]})
    new = pd.DataFrame({"sample_id": ["c", "b", "a"], "news_text": ["z", "y", "x"]})
    o, n = align(old, new)
    assert list(o["sample_id"]) == list(n["sample_id"]) == ["a", "b"]


def test_compare_runs_rejects_different_article_text_for_same_id():
    import pandas as pd
    import pytest
    from scripts.compare_runs import align
    with pytest.raises(ValueError):
        align(pd.DataFrame({"sample_id": ["a"], "news_text": ["x"]}), pd.DataFrame({"sample_id": ["a"], "news_text": ["other"]}))


def test_baselines_take_leading_sentences_and_words():
    from scripts.baselines import first_words, lead_sentences, mean_reference_words
    text = "Birinci cümle. İkinci cümle! Üçüncü cümle?"
    assert lead_sentences(text, 2) == "Birinci cümle. İkinci cümle!"
    assert first_words("a b c d", 2) == "a b"
    assert mean_reference_words(["a b c", "a b c d e"]) == 4


def test_shifted_references_never_return_the_own_reference():
    from scripts.baselines import shifted_references
    refs = ["r0", "r1", "r2"]
    shifted = shifted_references(refs)
    assert shifted == ["r1", "r2", "r0"] and all(a != b for a, b in zip(refs, shifted))


def test_decode_ablation_selects_only_among_base_variants_on_validation_rows():
    from scripts.decode_ablation import DECODING, base_rows, select_best
    rows = [{"variant": "a/s2", "rougeL": 40.0}, {"variant": "a/s2+repair", "rougeL": 45.0},
            {"variant": "b/s3", "rougeL": 41.5}, {"variant": "c/s2", "rougeL": 41.5}]
    assert select_best(base_rows(rows)) == "b/s3"
    assert DECODING["greedy-80"]["gen"] == {} and DECODING["greedy-80"]["max_new_tokens"] == 80
    assert all(cfg["gen"].get("num_beams", 1) >= cfg["candidates"] for cfg in DECODING.values())


def test_training_use_case_passes_validation_conversations_to_the_trainer():
    from application.use_cases import TrainLoraSummarizerUseCase
    from domain.entities import NewsArticle
    from domain.services.prompt_builder import PromptBuilder

    class FakeTrainer:
        def fine_tune(self, conversations, output_dir, eval_conversations=()):
            self.seen = (len(conversations), len(eval_conversations), output_dir)
            return output_dir

    trainer = FakeTrainer()
    arts = [NewsArticle(id=str(i), article_text="metin", reference_summary="özet") for i in range(3)]
    TrainLoraSummarizerUseCase(trainer, PromptBuilder()).execute(arts[:2], "out", arts[2:])
    assert trainer.seen == (2, 1, "out")
    TrainLoraSummarizerUseCase(trainer, PromptBuilder()).execute(arts, "out")
    assert trainer.seen == (3, 0, "out")
