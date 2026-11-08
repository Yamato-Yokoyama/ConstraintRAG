from src.eval_retrieval import trap_floor
from src.load_esci import make_document
from src.negation import extract_excluded_term, has_unnegated, is_negated


def test_detects_negated_query():
    assert is_negated("cat treats without chicken")
    assert not is_negated("cat treats")


def test_extracts_excluded_term():
    assert extract_excluded_term("alarm clock without snooze") == "snooze"
    assert extract_excluded_term("gluten-free bread") == "gluten"
    assert extract_excluded_term("cat treats") is None


def test_negated_mention_is_not_a_trap():
    assert has_unnegated("Chicken Cat Treats", "chicken")
    assert not has_unnegated("Cat Treats Without Chicken", "chicken")
    assert not has_unnegated("Chicken-Free Cat Treats", "chicken")


def test_trap_floor_counts_slots_clean_products_cannot_fill():
    assert trap_floor(["radio clock", "radio alarm", "plain clock"], "radio", k=3) == 2 / 3


def test_make_document_removes_html_and_nan():
    assert make_document("Clock", "<p>Quiet</p>") == "Clock\nQuiet"
    assert make_document("Clock", "nan") == "Clock"


def test_eval_queries_exist_in_esci():
    import pytest

    from src.load_esci import DATA_DIR, eval_query_ids, load_eval_queries, load_examples

    if not (DATA_DIR / "examples_us.parquet").exists():
        pytest.skip("ESCI data is not available")
    ids = eval_query_ids(load_eval_queries())
    assert set(ids) <= set(load_examples()["query_id"])
