from domain.services.summary_parser import SummaryParserService


def test_truncates_to_max_sentences():
    parser = SummaryParserService(max_sentences=2)
    assert parser.parse("Bir. İki. Üç. Dört.") == "Bir. İki."


def test_removes_repeated_sentences():
    parser = SummaryParserService(max_sentences=2)
    raw = "Bakan açıklama yaptı. Bakan açıklama yaptı. Bakan açıklama yaptı. Ek bilgi verildi."
    assert parser.parse(raw) == "Bakan açıklama yaptı. Ek bilgi verildi."


def test_strips_leading_summary_label():
    assert SummaryParserService().parse("Özet: Deprem oldu.") == "Deprem oldu."


def test_cot_keeps_only_text_after_final_marker():
    raw = "1. Ana olay: deprem\n2. Kişiler: yok\nNihai özet: Deprem sonrası hasar yok."
    assert SummaryParserService().parse(raw, chain_of_thought=True) == "Deprem sonrası hasar yok."


def test_cot_without_marker_drops_bullets():
    raw = "1. Ana olay: deprem\n- Kişiler: yok\nDeprem oldu."
    assert SummaryParserService().parse(raw, chain_of_thought=True) == "Deprem oldu."


def test_non_cot_output_keeps_list_like_lines():
    raw = "- Deprem oldu.\n- Kimse yaralanmadı."
    assert SummaryParserService(max_sentences=2).parse(raw) == "- Deprem oldu. - Kimse yaralanmadı."


def test_marker_matches_uppercase_dotted_i():
    raw = "Analiz burada.\nNİHAİ ÖZET: Deprem oldu."
    assert SummaryParserService().parse(raw, chain_of_thought=True) == "Deprem oldu."


def test_empty_input_returns_empty_string():
    assert SummaryParserService().parse("") == ""
    assert SummaryParserService().parse(None) == ""


def test_removes_chat_special_tokens():
    assert SummaryParserService().parse("Haber özeti.<|im_end|>") == "Haber özeti."


def test_does_not_split_after_initials():
    raw = "M.A. ve D.U. isimli şahıslar yakalandı. İkinci cümle. Üçüncü cümle."
    assert SummaryParserService(max_sentences=2).parse(raw) == "M.A. ve D.U. isimli şahıslar yakalandı. İkinci cümle."


def test_does_not_split_after_common_abbreviations():
    raw = "Dr. Ali Yılmaz konuştu. Prof. Veli geldi. Üçüncü."
    assert SummaryParserService(max_sentences=2).parse(raw) == "Dr. Ali Yılmaz konuştu. Prof. Veli geldi."


def test_strips_markdown_bold_after_final_marker():
    raw = "**Nihai özet:** Deprem oldu."
    assert SummaryParserService().parse(raw, chain_of_thought=True) == "Deprem oldu."


def test_strips_inline_markdown_asterisks():
    assert SummaryParserService().parse("**Deprem** oldu.") == "Deprem oldu."
