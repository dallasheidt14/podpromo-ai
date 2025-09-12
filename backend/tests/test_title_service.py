from backend.services.title_service import generate_titles

def test_titles_are_clean_and_unique():
    # Simulate noisy keywords observed in logs
    kws = [
        "brain store information",
        "saying picture speaks",
        "practical subjects not",
        "hopes goals dreams",
    ]
    titles = generate_titles(kws, used_titles=set(), limit=4)
    assert 1 <= len(titles) <= 4
    # Basic sanity: no "Next Information/Speaks/Not"
    bad = ("Next Information", "Next Speaks", "Next Not")
    for t in titles:
        for b in bad:
            assert b not in t
    # Deduped
    assert len(titles) == len(set(titles))
