from agent_memory.textsearch import matches, stem, text_score, tokenize


def test_tokenize_lowercases_and_drops_stop_words():
    assert tokenize("The Invoice is in the folder") == ["invoic", "folder"]


def test_stem_handles_plurals_and_tenses():
    assert stem("invoices") == stem("invoice") == "invoic"
    assert stem("scheduling") == stem("scheduled") == stem("schedule") == "schedul"
    assert stem("policies") == "policy"
    assert stem("bus") == "bus"


def test_no_overlap_scores_zero():
    assert text_score("quarterly tax filing", "Paint colours", "Use the warm white") == 0.0
    assert not matches("quarterly tax filing", "Paint colours", "Use the warm white")


def test_empty_or_stop_word_only_query_scores_zero():
    assert text_score("", "title", "content") == 0.0
    assert text_score("the and of", "title", "content") == 0.0


def test_score_is_bounded():
    score = text_score("deposit invoice", "Deposit invoice", "deposit invoice " * 200)
    assert 0.0 < score < 1.0


def test_more_query_terms_matched_scores_higher():
    partial = text_score("reschedule rain delay", "", "We reschedule on request")
    full = text_score("reschedule rain delay", "", "Reschedule after any rain delay")
    assert full > partial > 0


def test_title_match_outweighs_body_match():
    in_title = text_score("warranty", "Warranty terms", "Twelve months on labour")
    in_body = text_score("warranty", "Terms", "Twelve months warranty on labour")
    assert in_title > in_body


def test_matching_is_stem_insensitive():
    assert matches("invoices", "", "Send the invoice on completion")
