from app.classifier import Classifier


def fake_pipeline(comments, candidate_labels, multi_label):
    """Imitates HF zero-shot pipeline output shape."""
    # HF returns a dict when input is a single string, list of dicts when input is a list.
    # Classifier always passes a list, so we always return a list.
    return [
        {"labels": candidate_labels, "scores": [0.9, 0.05, 0.03, 0.02]}
        for _ in comments
    ]


def test_classify_batch_returns_top_label_and_score():
    c = Classifier(pipeline=fake_pipeline, candidate_labels=["A", "B", "C", "D"])
    results = c.classify_batch(["hi", "hello"])
    assert results == [("A", 0.9), ("A", 0.9)]


def test_classify_batch_empty():
    c = Classifier(pipeline=fake_pipeline, candidate_labels=["A", "B"])
    assert c.classify_batch([]) == []
