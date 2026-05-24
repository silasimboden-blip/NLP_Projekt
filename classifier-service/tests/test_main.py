import pytest
from fastapi.testclient import TestClient

from app.classifier import Classifier


@pytest.fixture
def client(monkeypatch):
    # Inject a fake classifier so we don't load the HF model.
    def fake_pipeline(comments, candidate_labels, multi_label):
        return [
            {"labels": candidate_labels, "scores": [0.8, 0.1, 0.05, 0.05]}
            for _ in comments
        ]

    from app import main
    main._classifier_factory = lambda: Classifier(
        pipeline=fake_pipeline,
        candidate_labels=["Zustimmung", "sachliche Kritik", "Empörung",
                          "Beleidigung oder persönlicher Angriff"],
    )
    # Build app via TestClient which triggers lifespan startup.
    with TestClient(main.app) as c:
        yield c


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_classify_returns_label_and_score(client):
    r = client.post("/classify", json={"comment": "Sehr guter Artikel!"})
    assert r.status_code == 200
    body = r.json()
    assert body["comment"] == "Sehr guter Artikel!"
    assert body["label"] == "Zustimmung"
    assert body["score"] == pytest.approx(0.8)


def test_classify_rejects_empty_comment(client):
    r = client.post("/classify", json={"comment": ""})
    assert r.status_code == 422


def test_metrics_endpoint_returns_prometheus_format(client):
    # Hit /classify once so there's data, then scrape.
    client.post("/classify", json={"comment": "Sehr guter Artikel!"})
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "comments_classified_total" in body
    assert "batches_processed_total" in body
    assert "microbatch_queue_size" in body
    assert 'comments_classified_by_label_total{label="Zustimmung"}' in body
