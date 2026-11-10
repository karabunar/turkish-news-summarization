import pytest

from domain.entities import Condition, NewsArticle
from domain.services.prompt_builder import PromptBuilder, split_few_shot_pool

EXAMPLES = [NewsArticle(str(i), f"Haber metni {i}.", f"Özet {i}.") for i in range(3)]


def test_c1_zero_shot_uses_exact_instruction():
    msgs = PromptBuilder().build(Condition.C1_ZERO_SHOT, "Metin")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert msgs[0]["content"].startswith("Aşağıdaki haber metnini 1-2 cümle ile anlaşılır şekilde özetle:")
    assert msgs[0]["content"].endswith("Metin")


def test_c2_few_shot_has_three_solved_examples():
    msgs = PromptBuilder(EXAMPLES).build(Condition.C2_FEW_SHOT, "Yeni haber")
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 3 + ["user"]
    assert msgs[1]["content"] == "Özet 0."
    assert msgs[-1]["content"].endswith("Yeni haber")


def test_c2_requires_exactly_three_examples():
    with pytest.raises(ValueError):
        PromptBuilder(EXAMPLES[:2]).build(Condition.C2_FEW_SHOT, "x")


def test_c3_contains_persona_and_final_marker():
    content = PromptBuilder().build(Condition.C3_COT, "Metin")[0]["content"]
    assert "Sen uzman bir haber editörüsün." in content
    assert "adım adım analiz et" in content and "Nihai özet:" in content


def test_c4_inference_prompt_equals_c1():
    b = PromptBuilder()
    assert b.build(Condition.C4_LORA, "Metin") == b.build(Condition.C1_ZERO_SHOT, "Metin")


def test_long_article_is_clipped():
    msgs = PromptBuilder(max_article_chars=50).build(Condition.C1_ZERO_SHOT, "kelime " * 100)
    assert len(msgs[0]["content"]) < 200


def test_clip_keeps_last_word_when_cut_is_on_a_word_boundary():
    b = PromptBuilder()
    assert b._clip("kelime kelime kelime", 13) == "kelime kelime"
    assert b._clip("kelime kelimexx", 11) == "kelime"
    assert b._clip("abcdefghijklmnop", 5) == "abcde"  # no space: cut mid-word


def test_few_shot_pool_split_is_disjoint_and_keeps_training_prefix():
    pool = [NewsArticle(str(i), "t", "s") for i in range(1003)]
    train, few = split_few_shot_pool(pool, 1000)
    assert [a.id for a in train] == [str(i) for i in range(1000)]
    assert [a.id for a in few] == ["1000", "1001", "1002"]
    with pytest.raises(ValueError):
        split_few_shot_pool(pool[:10], 1000)


def test_training_conversation_ends_with_reference_summary():
    conv = PromptBuilder().build_training_conversation(NewsArticle("1", "Metin", "Referans özet."))
    assert conv[-1] == {"role": "assistant", "content": "Referans özet."}
