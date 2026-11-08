from src.compare_retrievers import reciprocal_rank_fusion


def test_rrf_rewards_documents_in_both_lists():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]])[0] == "b"
