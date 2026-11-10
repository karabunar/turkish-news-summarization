from domain.services.entity_repair import pick_most_faithful, repair_entities


def test_repairs_misspelled_entity_with_similar_article_word_and_keeps_suffix():
    article = "Başkan Volkan Konak dün İstanbul'da açıklama yaptı."
    assert repair_entities("Başkan Volkon Konak açıklama yaptı.", article) == "Başkan Volkan Konak açıklama yaptı."
    assert repair_entities("Açıklama İstanbul'da yapıldı, Volkon’un sözleri.", article) == \
        "Açıklama İstanbul'da yapıldı, Volkan’un sözleri."


def test_leaves_supported_and_dissimilar_entities_and_numbers_alone():
    article = "Başkan Volkan Konak dün İstanbul'da 25 kişiyle görüştü."
    text = "Ankara'da 30 kişi Volkan ile görüştü."
    assert repair_entities(text, article) == text


def test_similarity_threshold_controls_how_aggressive_the_repair_is():
    article = "Volkan Konak açıklama yaptı."
    assert repair_entities("Volkon açıklama yaptı.", article, min_similarity=0.95) == "Volkon açıklama yaptı."


def test_pick_most_faithful_prefers_fewest_unsupported_entities_then_first():
    article = "Ali Veli İstanbul'a gitti."
    cands = ["Mehmet Ankara'ya gitti.", "Ali İstanbul'a gitti.", "Veli İstanbul'a gitti."]
    assert pick_most_faithful(cands, article) == "Ali İstanbul'a gitti."
    assert pick_most_faithful(["x y", "z"], article) == "x y"
