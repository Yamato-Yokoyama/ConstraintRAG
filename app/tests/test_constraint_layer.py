from types import SimpleNamespace

from src.constraint_layer import captions_db, filter_candidates, parse_constraints


def test_exclusion_removes_only_matching_captions_and_keeps_order():
    connection = captions_db({"a": "a chicken snack", "b": "a fish snack"})
    assert filter_candidates(connection, ["c", "a", "b"], {"include": [], "exclude": ["chicken"]}) == ["c", "b"]


def test_term_cannot_inject_sql():
    connection = captions_db({"a": "snack"})
    assert filter_candidates(connection, ["a"], {"include": [], "exclude": ["x'; DROP TABLE captions; --"]}) == ["a"]


def test_parse_constraints_reads_fenced_json():
    llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(text='```json\n{"include":["cat"],"exclude":["chicken"]}\n```'))
    assert parse_constraints("cat treats without chicken", llm)["exclude"] == ["chicken"]
